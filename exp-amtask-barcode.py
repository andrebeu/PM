import sys

import torch as tr
import numpy as np

from PM_models import *
from PM_tasks import *

GPU = tr.cuda.is_available()

seed = int(sys.argv[1])
instdim = int(sys.argv[2])
stimdim = int(sys.argv[3])
wmsize = int(sys.argv[4])

nmaps = 4
emsetting = 1
switch = 1
ntokens = 0

fdir = 'model_data/amtask-barcode/'
fname = 'nmaps_%i-ntrials_%i-wmsize_%i-instdim_%i-stimdim_%i-seed_%i'%(
            nmaps,ntrials,wmsize,instdim,stimdim,seed)

net = NetBarCode(wmsize=wmsize,
                 emsetting=1,
                 instdim=instdim,
                 stimdim=stimdim,
                 seed=seed,
                 debug=False)

if GPU: net.cuda()

task = TaskArbitraryMaps(
        nmaps=nmaps,
        switchmaps=switch,
        ntokens_surplus=ntokens,
        seed=seed,
        stimdim=stimdim
        )

maxsoftmax = lambda ulog: tr.argmax(tr.softmax(ulog,-1),-1)

def run_net(net,task,neps,ntrials,trlen,training=True):
  '''
  returns score [neps,ntrials,nmaps+trlen]
  '''
  lossop = tr.nn.CrossEntropyLoss()
  optiop = tr.optim.Adam(net.parameters(), lr=0.001)
  exp_len = ntrials*(task.nmaps+trlen)
  score = -np.ones([neps,exp_len])
  for ep in range(neps):
    # forward prop
    iseq,xseq,ytarget = task.gen_ep_data(ntrials,trlen)
    if GPU:
      iseq = iseq.cuda()
      xseq = xseq.cuda()
      ytarget = ytarget.cuda()
    yhat_ulog = net(iseq,xseq)
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
    if ep%(neps/5)==0:
      print(ep/neps,score_t.mean())
  score = score.reshape(neps,ntrials,trlen+task.nmaps)
  return score


## curriculum 
print('TRAIN')
trscL = []
nblocks = 2
emkL = ['stim','conj']
nepsL = [2000,10000]
ntrials = 2
trlen = 1
for idx in range(nblocks):
  net.emk=emkL[idx]
  trsc = run_net(net,task,nepsL[idx],ntrials,trlen,training=True)
  trscL.append(trsc)

## save
trsc = np.concatenate(trscL)
np.save(fdir+fname+'-trsc',trsc)
tr.save(net.state_dict(),fdir+fname+'-model.pt')

print('EVAL')
neps_ev = 500
ntrials_ev = 15
trlen_ev = 5

for em in [1,0]:
  net.EMsetting=em
  evsc = run_net(net,task,neps_ev,ntrials_ev,trlen_ev,training=False)
  np.save(fdir+fname+'-ev_em_%i-evsc'%em,evsc)