import torch as tr
import numpy as np

import itertools

from PM_models import *
from PM_tasks import *

from scipy.stats import pearsonr
from scipy.spatial import distance


def mov_avg(arr,wind):
  MA = -np.ones(len(arr)-wind)
  for t in range(len(arr)-wind):
    MA[t] = arr[t:t+wind].mean()
  return MA

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

### RDM package


dist_metric = 'cosine'
compute_rdm = lambda M: distance.cdist(M,M,metric=dist_metric)

def get_mean_sub_rdm(states):
  ''' loop over epochs '''
  neps,tsteps,stsize = states.shape
  rdm = np.zeros([tsteps,tsteps])
  for ep in range(neps):
    rdm += compute_rdm(states[ep])
  rdm = rdm/neps
  return rdm

def compute_rdms(states):
  """ 
  loop over nets and epochs 
  given cstates [nnets,neps,tsteps,stsize]
  computes TxT rsm for each subj (averaged over neps)
  returns rsm per net [nnets,tsteps,tsteps]
  """

  # init 
  nnets,neps,tsteps,stsize = states.shape
  rsms = -np.ones([nnets,tsteps,tsteps])
  # loop compute RSM
  for sub in range(nnets):
    sub_rsms = -np.ones([neps,tsteps,tsteps])
    sub_state = states[sub]
    for ep in range(neps):
      sub_rsms[ep] = compute_rdm(sub_state[ep])
    rsms[sub] = sub_rsms.mean(0)
  return rsms

def get_rdms_full(states):
  ''' takes rdsm [nnets,neps,ntrials,seqlen,stsize]
  returns [nnets,ntrials*seqlen,ntrials*seqlen]'''
  nnets,neps,ntrials,seqlen,stsize = states.shape
  states_full = np.concatenate([states[:,:,i,:,:] for i in range(ntrials)],-2)
  return compute_rdms(states_full)



## load net for amtask-sweep1 

# def load_net(emsetting,ntrials,trlen,ntoksurp,seed):
#   stsize,nmaps = 20,4
#   net = NetAMEM(stsize=stsize,emsetting=emsetting,wmsetting=1,seed=seed)
#   fdir = 'model_data/amtask-sweep1/'
#   fpath = 'lstm_%i-em_%s-nmaps_%s-ntrials_%s-trlen_%s-ntoksurp_%i-seed_%s-model.pt'%(
#             stsize,emsetting,nmaps,ntrials,trlen,ntoksurp,seed)
#   try:
#     net.load_state_dict(tr.load(fdir+fpath))
#     net.seed = seed
#     net.ntoksurp = ntoksurp
#     net.ntrials = ntrials
#     net.trlen = trlen
#     net.em_train = emsetting
#   except:
#     print('fail to load',fpath)
#     return None
#   return net
  
# def load_netL(emsetting,ntrials,trlen,ntoksurp,nnets=20):
#   netL = []
#   for seed in range(nnets):
#     net = load_net(emsetting,ntrials,trlen,ntoksurp,seed)
#     if net==None: 
#       continue
#     netL.append(net)
#   print('N =',len(netL))
#   return netL

# def eval_net(net,task,neps,ntrials,trlen):
#   '''
#   returns score [neps,ntrials,nmaps+trlen]
#   '''
#   lossop = tr.nn.CrossEntropyLoss()
#   optiop = tr.optim.Adam(net.parameters(), lr=0.001)
#   exp_len = ntrials*(task.nmaps+trlen)
#   score = -np.ones([neps,exp_len])
#   for ep in range(neps):
#     # forward prop
#     iseq,xseq,ytarget = task.gen_ep_data(ntrials,trlen)
#     yhat_ulog = net(iseq,xseq)
#     # eval
#     score_t = (maxsoftmax(yhat_ulog) == ytarget).numpy()
#     score[ep] = np.squeeze(score_t)
#   score = score.reshape(neps,ntrials,trlen+task.nmaps)
#   return score

# def eval_and_save(net,neps,ntr,trlen):
#   nmaps=4
#   save_fdir = 'model_data/amtask-sweep1_analysis/eval/'
#   for em_ev in [0,1]:
#     net.EMsetting=em_ev
#     task = TaskArbitraryMaps(nmaps=4,switchmaps=1,ntokens_surplus=net.ntoksurp,seed=net.seed)
#     evacc = eval_net(net,task,neps,ntr,trlen)
#     fpath = 'lstm_20-em_%s-nmaps_%s-ntrials_%s-trlen_%s-ntoksurp_%s-seed_%s-em_ev_%i-evsc.npy'%(
#               net.em_train,nmaps,net.ntrials,net.trlen,net.ntoksurp,net.seed,em_ev)
#     np.save(save_fdir+fpath,evacc)
