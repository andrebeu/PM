import sys

import torch as tr
import numpy as np

from PM_models import *
from PM_tasks import *

GPU = tr.cuda.is_available()

seed = int(sys.argv[1])
instdim = int(sys.argv[2])
stimdim = int(sys.argv[3])
stsize = int(sys.argv[4])
deep = int(sys.argv[5])
wmsetting = int(sys.argv[6])

# task
switchmaps = 1
nmaps = 2
ntokens_surplus = 0
# net
emsetting = 0
# training
ntrials = 1
trlen = 1
neps = 100000

fdir = 'model_data/amtask-mini/'
fname = 'lstm(%i)_%i-deep_%i-em_%i-instdim_%i-stimdim_%i-seed_%i'%(
            wmsetting,stsize,deep,emsetting,instdim,stimdim,seed)

task = TaskArbitraryMaps(
  nmaps=nmaps,
  switchmaps=switchmaps,
  ntokens_surplus=0,
  seed=seed,
  stimdim=stimdim
  )
net = NetAMEM(stsize=stsize,
  emsetting=emsetting,
  wmsetting=wmsetting,
  seed=seed,
  instdim=instdim,
  stimdim=stimdim
  )
net.deep = deep

if GPU:
  print('GPU')
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

print('TRAIN')
trsc = run_net(net,task,neps,ntrials,trlen,training=True)
np.save(fdir+fname+'-trsc',trsc)
tr.save(net.state_dict(),fdir+fname+'-model.pt')

