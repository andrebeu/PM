from glob import glob as glob
from itertools import product

import torch as tr
import numpy as np

from PM_models import *
from PM_tasks import *



def load_net(seed,trep,switch,ntokens,seqlen,ntrials):
  """ 
  """
  ##
  model_dir = 'PITask_randstim' 
  model_name = 'LSTM_25-EM_conjcode'
  trep *= 100000
  fpath = 'model_data/%s/'%model_dir
  fpath += "%s-ntokens_%i-seqlen_%i-ntrials_%i-switchmaps_%s-seed_%s-tr_%s-model.pt"%(
              model_name,ntokens,seqlen,ntrials,switch,seed,trep)
  # load
  net = PINet(stimdim=10,stsize=25,outdim=ntokens,nmaps=ntokens+1,seed=seed)
  net.load_state_dict(tr.load(fpath))
  return net

def load_netL(trep,switch,ntokens,seqlen,ntrials,nnets=10):
  """ loop load_net for every seed
  """
  netL = []
  for sd in range(nnets):
    try:
      net = load_net(sd,trep,switch,ntokens,seqlen,ntrials)
      netL.append(net)
    except:
      pass
  print('N =',len(netL),'loaded nets')
  return netL


def eval_net(net,task,neps,ntrials,seqlen,switch):
  """ 
  fixed sequence length eval
  new embedding for every episode
  returns score
  """
  maxsoftmax = lambda ulog: tr.argmax(tr.softmax(ulog,-1),-1)
  tsteps = ntrials*(seqlen+task.ntokens)
  score = -np.ones((neps,tsteps))
  cstates = -np.ones((neps,tsteps,net.stsize))
  for ep in range(neps):
    task.sample_emat()
    tseq,xseq,ytarget = task.gen_ep_data(ntrials,seqlen,switch)
    yhat_ulog = net(tseq,xseq)
    cstates[ep] = net.cstates
    score[ep] = (maxsoftmax(yhat_ulog)==ytarget).squeeze()
  return score,cstates

def eval_netL(netL,embool,task,neps,ntrials=10,seqlen=10,switch=1):
  """ returns acc `[sub,tsteps]`
  """
  nsubs = len(netL)
  tsteps = ntrials*(seqlen + task.ntokens)
  group_acc = -np.ones([nsubs,tsteps]) 
  group_cstates = -np.ones([nsubs,neps,tsteps,netL[0].stsize])
  for sid,net in enumerate(netL):
    net.EMbool = embool
    evsc,cstates = eval_net(net,task,neps,ntrials,seqlen,switch)
    group_acc[sid] = evsc.mean(0)
    group_cstates[sid] = cstates
  return group_acc,group_cstates


## train params
ntokens = 10
trepL = [9,5]
switchL = [0,1]
seqlenL = [5,10,13,15,20]
ntrialsL = [2,3,4,10]
nnets = 10

## eval params
ev_neps = 500  
ev_ntrials = 11
ev_seqlen = 5
ev_switch = 1
ev_EML = [0,1]

## init task
task = PurePM(ntokens=ntokens,stimdim=10,seed=np.random.randint(999)) 

## main loop: eval, save data
for trep,seqlen,ntrials,switch in product(trepL,seqlenL,ntrialsL,switchL):
  print('trep',trep,'switch',switch,'seqlen',seqlen,'ntrials',ntrials)
  netL = load_netL(trep,switch,ntokens,seqlen,ntrials,nnets=nnets)
  if not len(netL): continue
  for ev_EM in ev_EML:
    # eval data
    acc,cstates = eval_netL(netL,ev_EM,task,ev_neps,ntrials=ev_ntrials,seqlen=ev_seqlen,switch=ev_switch)
    # save
    save_fpath = "LSTM25-EMcc-ntokens10-trep_%i-tr_switch_%i-tr_seqlen_%i-tr_ntrials_%i-ev_EM_%i"%(
                    trep,switch,seqlen,ntrials,ev_EM)
    np.save('model_data/buckets_eval_data/'+save_fpath+'-evacc',acc)
    np.save('model_data/buckets_eval_data/'+save_fpath+'-cstates',cstates)