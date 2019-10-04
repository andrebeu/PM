import sys

import torch as tr
import numpy as np

from PM_models import *
from PM_tasks import *



seed = int(sys.argv[1])
# task
switchmaps = 1
nmaps = 4
ntokens_surplus = 0
# net
stsize = 20
wmsetting = 1
emsetting = int(sys.argv[2])
# training
ntrials = int(sys.argv[3])
trlen = int(sys.argv[4])
neps = 150000

fdir = 'model_data/amtask-sweep1/'
fname = 'lstm_%i-em_%i-nmaps_%i-ntrials_%i-trlen_%i-ntoksurp_%i-seed_%i'%(
            stsize,emsetting,nmaps,ntrials,trlen,ntokens_surplus,seed)

task = TaskArbitraryMaps(nmaps=nmaps,
                         switchmaps=switchmaps,
                         ntokens_surplus=ntokens_surplus,
                         seed=seed)

net = NetAMEM(stsize=stsize,
              emsetting=emsetting,
              wmsetting=wmsetting,
              seed=seed)


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
  score = score.reshape(neps,ntrials,trlen+task.nmaps)
  return score

trsc = run_net(net,task,neps,ntrials,trlen,training=True)
np.save(fdir+fname+'-trsc',trsc)
tr.save(net.state_dict(),fdir+fname+'-model.pt')