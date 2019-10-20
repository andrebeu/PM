import sys

import torch as tr
import numpy as np

from PM_models import *
from PM_tasks import *


wmsize = int(sys.argv[1])
nmaps = int(sys.argv[2])
switch = int(sys.argv[3])
ntrials = int(sys.argv[4])
trlen_tr = int(sys.argv[5])
seed = int(sys.argv[6])

# net params
emsetting = 1
instdim = 10
stimdim = 12
emk_weights = [1,.0005]

# train params
neps_tr = 50000 

fdir = 'model_data/amtask_pieval/'

fpath = 'wmsize_%i-nmaps_%i-switch_%i-ntrials_%i-trlen_%i-seed_%i-wm_emkw_%f'%(
          wmsize,nmaps,switch,ntrials,trlen_tr,seed,emk_weights[1])
print(fpath)

net = NetBarCode(
        wmsize=wmsize,
        emsetting=emsetting,
        seed=seed,
        instdim=instdim,
        stimdim=stimdim,
        init_emkw=emk_weights,
        debug=False
)

if tr.cuda.is_available():
  net.cuda()

task = TaskArbitraryMaps(
          nmaps=nmaps,
          switchmaps=switch,
          ntokens_surplus=0,
          seed=seed,
          stimdim=stimdim)

maxsoftmax = lambda ulog: tr.argmax(tr.softmax(ulog,-1),-1)

def run_net(net,task,neps,ntrials,trlen,training=True,verb=True,return_states=False):
  '''
  returns score [neps,ntrials,nmaps+trlen]
  '''
  lossop = tr.nn.CrossEntropyLoss()
  optiop = tr.optim.Adam(net.parameters(), lr=0.001)
  exp_len = ntrials*(task.nmaps+trlen)
  score = -np.ones([neps,exp_len])
  states = -np.ones([neps,exp_len,2,net.wmdim])
  net.store_states = return_states
  for ep in range(neps):
    # forward prop
    iseq,xseq,ytarget = task.gen_ep_data(ntrials,trlen)
    if tr.cuda.is_available():
      iseq = iseq.cuda()
      xseq = xseq.cuda()
      ytarget = ytarget.cuda()
    yhat_ulog = net(iseq,xseq)
    if return_states:
      states_ep = net.states
      states[ep] = net.states
    # eval
    score_t = (maxsoftmax(yhat_ulog) == ytarget).cpu().numpy()
    score[ep] = np.squeeze(score_t)
    if training:
      # backprop
      loss = 0
      for tstep in range(len(iseq)):
        loss += lossop(yhat_ulog[tstep],ytarget[tstep])
      optiop.zero_grad()
      loss.backward(retain_graph=True)
      optiop.step()
    if verb and ep%(neps/5)==0:
      print(ep/neps,score_t.mean())
  score = score.reshape(neps,ntrials,trlen+task.nmaps)
  if return_states:
    return score,states
  return score


trsc = run_net(net,task,neps_tr,ntrials,trlen_tr,
                training=True,verb=True,return_states=False)

np.save(fdir+fpath+'-trsc',trsc)


task = TaskArbitraryMaps(
          nmaps=nmaps,
          switchmaps=1,
          ntokens_surplus=0,
          seed=seed,
          stimdim=stimdim)

neps_ev = 1000
ntrials_ev = 20
trlen_ev = 10

evsc,states = run_net(net,task,neps_ev,ntrials_ev,trlen_ev,
                training=True,verb=True,return_states=True)

np.save(fdir+fpath+'-evsc',evsc)
np.save(fdir+fpath+'-states_ev',states)

