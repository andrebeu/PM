import sys

import torch as tr
import numpy as np

from PM_models import *
from PM_tasks import *
# from help_amtask import *

wmsize = int(sys.argv[1])
nmaps = int(sys.argv[2])
switch = int(sys.argv[3])
ntrials = int(sys.argv[4])
seed = int(sys.argv[5])

emsetting = 1
instdim = 10
stimdim = 12

emk_weights = [1,.0005]

fpath = 'wmsize_%i-nmaps_%i-switch_%i-ntrials_%i-seed_%i'%(wmsize,nmaps,switch,ntrials,seed)
print(fpath)

net = NetBarCode(
        wmsize=wmsize,
        emsetting=emsetting,
        seed=seed,
        instdim=instdim,
        stimdim=stimdim,
        debug=False)

net.emk_weights = emk_weights  # stim,lstm

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
  for ep in range(neps):
    # forward prop
    iseq,xseq,ytarget = task.gen_ep_data(ntrials,trlen)
    if tr.cuda.is_available():
      iseq = iseq.cuda()
      xseq = xseq.cuda()
      ytarget = ytarget.cuda()
    yhat_ulog = net(iseq,xseq)
    if net.store_states:
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


neps_tr = 50000

trlen_tr = 1
task.switchmaps = True
trsc = run_net(net,task,neps_tr,ntrials,trlen_tr,training=True,verb=True)

np.save('model_data/amtask-barcode2/'+fpath,trsc)


