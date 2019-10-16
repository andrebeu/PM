import sys

import torch as tr
import numpy as np

from PM_models import *
from PM_tasks import *

seed = int(sys.argv[1])
nmaps = int(sys.argv[2])
ntrials = int(sys.argv[3])
switch = int(sys.argv[4])

instdim = 10
stimdim = 12
wmsize = 6

emsetting = 1

ntokens = 0
## defines training curriculum
nepsL = [1000,39000]
curr = '_'.join([str(i) for i in nepsL])

fdir = 'model_data/amtask-barcode/'
fname = 'nmaps_%i-ntrials_%i-switch_%i-wmsize_%i-instdim_%i-stimdim_%i-curr_%s-seed_%i'%(
            nmaps,ntrials,switch,wmsize,instdim,stimdim,curr,seed)

net = NetBarCode(wmsize=wmsize,
                 emsetting=1,
                 instdim=instdim,
                 stimdim=stimdim,
                 seed=seed,
                 debug=False)

if tr.cuda.is_available(): net.cuda()

task = TaskArbitraryMaps(
        nmaps=nmaps,
        switchmaps=switch,
        ntokens_surplus=ntokens,
        seed=seed,
        stimdim=stimdim
        )

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


 
## curriculum training loop
print('TRAIN')
emkL = ['stim','conj']
trlen = 1
trscL = []
for idx in range(len(emkL)):
  net.emk=emkL[idx]
  trsc = run_net(net,task,nepsL[idx],ntrials,trlen,training=True)
  trscL.append(trsc)

## save train data
trsc = np.concatenate(trscL)
np.save(fdir+fname+'-trsc',trsc)
tr.save(net.state_dict(),fdir+fname+'-model.pt')

## eval
print('EVAL')
task.switchmaps=True
net.emk='conj'
net.store_states=True
neps_ev = 500
ntrials_ev = 15
trlen_ev = 5

for em in [0,1]:
  net.EMsetting=em
  evsc,states = run_net(net,task,neps_ev,ntrials_ev,trlen_ev,training=False,return_states=True)
  np.save(fdir+fname+'-ev_em_%i-evsc'%em,evsc)
np.save(fdir+fname+'-ev_em_%i-states'%em,states)