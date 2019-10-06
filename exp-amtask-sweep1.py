import sys

import torch as tr
import numpy as np

from PM_models import *
from PM_tasks import *

GPU = True

seed = int(sys.argv[1])
# task
switchmaps = 1
nmaps = 4
ntokens_surplus = 50
# net
stsize = 20
wmsetting = 1
emsetting = int(sys.argv[2])
# training
ntrials = int(sys.argv[3])
trlen = int(sys.argv[4])
neps = 100

fdir = 'model_data/amtask-sweep1/'
fname = '_GPU_TEST-lstm_%i-em_%i-nmaps_%i-ntrials_%i-trlen_%i-ntoksurp_%i-seed_%i'%(
            stsize,emsetting,nmaps,ntrials,trlen,ntokens_surplus,seed)

task = TaskArbitraryMaps(nmaps=nmaps,
                         switchmaps=switchmaps,
                         ntokens_surplus=ntokens_surplus,
                         seed=seed)

net = NetAMEM(stsize=stsize,
              emsetting=emsetting,
              wmsetting=wmsetting,
              seed=seed)
if GPU:
  net.cuda()

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
      iseq.cuda()
      xseq.cuda()
      ytarget.cuda()
    yhat_ulog = net(iseq,xseq)
    # eval
    score_t = (maxsoftmax(yhat_ulog) == ytarget).numpy()
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

## train
trsc = run_net(net,task,neps,ntrials,trlen,training=True)

## eval
neps_ev = 10
ntrials_ev = 2
trlen_ev = 5

net.EMsetting = 1
evsc_em1 = run_net(net,task,neps_ev,ntrials_ev,trlen_ev,training=False)
net.EMsetting = 0
evsc_em0 = run_net(net,task,neps_ev,ntrials_ev,trlen_ev,training=False)

## save
np.save(fdir+fname+'-trsc',trsc)
np.save(fdir+fname+'em_ev_1-evsc',evsc_em1)
np.save(fdir+fname+'em_ev_0-trsc',evsc_em0)
tr.save(net.state_dict(),fdir+fname+'-model.pt')