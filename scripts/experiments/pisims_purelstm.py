import sys
import torch as tr
import numpy as np

from PM_models import *
from PM_tasks import *

## params
# net
seed = int(sys.argv[1])
stsize = int(sys.argv[2])
sdim = 10
# task
ntokens = int(sys.argv[3])
seqlen = int(sys.argv[4])
ntrials = 2
switchmaps = 0
# epochs
neps_tr = 100000
neps_ev = 1000

model_fname = "LSTM_%i-ntokens_%i-seqlen_%i-ntrials_%i-switchmaps_%i-seed_%i"%(
                stsize,ntokens,seqlen,ntrials,switchmaps,seed)
print(model_fname)

task = PurePM(
        ntokens=ntokens,
        stimdim=sdim,
        seed=seed
)

net = PINet(
        stimdim=sdim,
        stsize=stsize,
        outdim=ntokens,
        ninstructs=ntokens+1,
        seed=seed
)


## funs
maxsoftmax = lambda ulog: tr.argmax(tr.softmax(ulog,-1),-1)
                           
def train_model(net,task,neps,ntrials,seqlen,switchmaps):
  """ 
  variable sequence length training
  closed loop randomizing of embedding
  """
  lossop = tr.nn.CrossEntropyLoss()
  optiop = tr.optim.Adam(net.parameters(), lr=0.001)
  acc = -np.ones(neps)
  for ep in range(neps):
    # forward prop
    tseq,xseq,ytarget = task.gen_ep_data(ntrials,seqlen=seqlen,switchmaps=switchmaps)
    yhat_ulog = net(tseq,xseq)
    # eval
    trial_acc = np.mean((maxsoftmax(yhat_ulog) == ytarget).numpy())
    acc[ep] = trial_acc
    # backprop
    for tstep in range(len(tseq)):
      optiop.zero_grad()
      loss = lossop(yhat_ulog[tstep],ytarget[tstep])
      loss.backward(retain_graph=True)
      optiop.step()
    if trial_acc>=.99:
      task.randomize_emat()
  return acc

def eval_model(net,task,neps,ntrials,seqlen,switchmaps):
  """ 
  fixed sequence length eval
  new embedding for every episode
  """
  score = -np.ones((neps,ntrials*(seqlen+task.ntokens)))
  for ep in range(neps):
    task.randomize_emat()
    tseq,xseq,ytarget = task.gen_ep_data(ntrials,seqlen,switchmaps)
    yhat_ulog = net(tseq,xseq)
    score[ep] = (maxsoftmax(yhat_ulog) == ytarget).squeeze()
  return score


## train and eval
tr_acc = train_model(net,task,neps_tr,ntrials,seqlen,switchmaps)
ev_score = eval_model(net,task,neps_ev,ntrials,seqlen,switchmaps)

fpath = "model_data/PITask/"+model_fname
np.save(fpath+"-tracc",tr_acc)
np.save(fpath+"-evscore",ev_score)
tr.save(net.state_dict(),fpath+'-model.pt')