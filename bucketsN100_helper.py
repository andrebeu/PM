from glob import glob as glob
import itertools

import torch as tr
import numpy as np

from scipy.stats import pearsonr
from scipy.spatial import distance


maxsoftmax = lambda ulog: tr.argmax(tr.softmax(ulog,-1),-1)

## RDM



## eval net

def load_netL(switch,trep=9,nnets=10,embool=1):
  netL = []
  for seed in range(nnets):
    fpath = "model_data/PITask_randstim/"
    fpath += "LSTM_25-EM_conjcode-ntokens_10-seqlen_13-ntrials_2-switchmaps_%i-seed_%i-tr_%i00000"%(switch,seed,trep)
    net = PINet(stimdim=10,stsize=25,outdim=10,nmaps=11,seed=seed)
    net.load_state_dict(tr.load(fpath+'-model.pt'))
    net.seed = seed
    net.EMbool = embool
    netL.append(net)
  return netL

def eval_net(net,gen_data_fn=None,neps=500,return_states=False):
  i,s,a = gen_data_fn()
  exp_len = len(i)
  score = -np.ones([neps,exp_len])
  states = -np.ones([neps,2,exp_len,net.stsize]) # hstate,cstate
  for ep in range(neps):
    # forward prop
    I,S,atarget = gen_data_fn()
    ahat_ulog = net(I,S)
    states[ep] = np.stack([net.hstates,net.cstates])
    trial_score = (maxsoftmax(ahat_ulog) == atarget).numpy()
    score[ep] = trial_score.squeeze()
  ev_acc = score.mean(0)
  if not return_states:
    return ev_acc
  else:
    return ev_acc,states

def eval_netL(netL,ntrials,seqlen,neps=10,return_states=True):
  """ states shape: [nsubs,neps,(h,c),tsteps,stsize]
  """
  nnets = len(netL)
  acc = []
  states = []
  for net in netL:
    task = PurePM(ntokens=10,stimdim=10,seed=net.seed)
    gen_data_fn = lambda: task.gen_ep_data(ntrials,seqlen,switchmaps=1)
    acc_net,states_net = eval_net(net,gen_data_fn,neps,return_states)
    acc.append(acc_net)
    states.append(states_net)
  acc = np.stack(acc)
  states = np.stack(states)
  return acc,states

## load save

def load_em_data():
  '''load presaved eval states and acc'''
  ntrials,seqlen = 6,5
  fpath = 'model_data/buckets_eval_data_N100/ntrials_%i-seqlen_%i'%(ntrials,seqlen)
  acc = np.load(fpath+'-acc.npy')
  states = np.load(fpath+'-states.npy')

  _,nnets,_ = acc.shape
  _,_,neps,_,_,_ = states.shape
  # print(states.shape) # (switch,nnets,eps,(h,c),tsteps,stsize)
  # print(acc.shape) # (switch,nnets,tsteps)

  '''reshape to remove instruction phase, select c or h state, and pull trials dimension'''
  # acc
  acc = acc.reshape(2,nnets,ntrials,10+seqlen)[:,:,:,10:]
  acc0,acc1 = acc

  # state
  state_class = 'hstate'
  states = states[:,:,:,int(state_class=='cstate'),:,:] # switch,nnets,neps,(h,c),tsteps,stsize
  states = states.reshape(2,nnets,neps,ntrials,10+seqlen,25)[:,:,:,:,10:,:]

  '''rename for comparison with lstms'''
  print('(switch,nnets,neps,ntrials,seqlen,stsize)')
  print(states.shape)
  print('(switch,nnets,ntrials,seqlen)')
  print(acc.shape)
  return states,acc

def load_lstm_data():
  nnets = 100
  ntrials,seqlen = 6,5
  neps = 200
  ''' load acc '''
  fpath = 'model_data/buckets_eval_data_N100/ntrials_%i-seqlen_%i-pure_lstm'%(ntrials,seqlen)
  acc = np.load(fpath+'-acc.npy')
  acc = acc.reshape(2,nnets,ntrials,10+seqlen)[:,:,:,10:]
  ''' load states '''
  states = np.load(fpath+'-states.npy')
  states = states[:,:,:,0,:,:] 
  states = states.reshape(2,nnets,neps,ntrials,10+seqlen,25)[:,:,:,:,10:]
  print(states.shape)
  ''' append _lstms to name '''
  acc_lstm = acc
  states_lstm = states
  return states,acc

def train_eval_save_lstm():
  ''' eval and save acc and states'''
  nnets = 100
  ntrials,seqlen = 6,5
  neps = 200

  accL = []
  statesL = []
  for switch in [0,1]:
    netL = load_netL_lstms(nnets,switch=switch)
    acc,states = eval_netL(netL,ntrials,seqlen,neps=neps)
    accL.append(acc)
    statesL.append(states)

  acc = np.stack(accL)
  states = np.stack(statesL)

  fpath = 'model_data/buckets_eval_data_N100/ntrials_%i-seqlen_%i-pure_lstm'%(ntrials,seqlen)
  np.save(fpath+'-acc',acc)
  np.save(fpath+'-states',states)
  return None

def train_eval_save_em():
  ''' eval and save acc and states of both groups'''
  nnets = 100
  ntrials,seqlen = 6,5
  neps = 200

  accL = []
  statesL = []
  for s in range(2):
    netL = load_netL(switch=s,nnets=nnets)
    acc,states = eval_netL(netL,ntrials,seqlen,neps=neps)
    accL.append(acc)
    statesL.append(states)

  acc = np.stack(accL)
  states = np.stack(statesL)

  fpath = 'model_data/buckets_eval_data_N100/ntrials_%i-seqlen_%i'%(ntrials,seqlen)
  np.save(fpath+'-acc',acc)
  np.save(fpath+'-states',states)
  return None


