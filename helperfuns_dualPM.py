from PM_models import *
from PM_tasks import *
import torch as tr
import numpy as np

from sklearn.metrics import pairwise_distances


maxsoftmax = lambda ulog: tr.argmax(tr.softmax(ulog,-1),-1)


def train_net(net,task,neps,gen_data_fn=None,verb=False):
  if type(gen_data_fn)==type(None):
    gen_data_fn = lambda : task.gen_ep_data(
                    num_trials=1,
                    trial_len=30,
                    pos_og_bias=np.random.randint(1,100,1),
                    pm_probes_per_trial=np.random.randint(0,10,1)
                    )
  lossop = tr.nn.CrossEntropyLoss()
  optiop = tr.optim.Adam(net.parameters(), lr=0.001)
  acc = -np.ones(neps)
  for ep in range(neps):
    # generate data
    I,S,atarget = gen_data_fn()
    # forward prop
    ahat_ulog = net(I,S)
    # eval compute acc
    trial_score = (maxsoftmax(ahat_ulog) == atarget).numpy()
    acc[ep] = trial_score.mean()
    # backprop
    loss = 0
    optiop.zero_grad()
    for tstep in range(len(atarget)):
      loss += lossop(ahat_ulog[tstep],atarget[tstep])
    loss.backward(retain_graph=True)
    optiop.step()
    if verb:
      if ep%(neps/5)==0:
        print(ep/neps,loss.detach().numpy())
  return acc

def eval_net2(net,task,neps=500,num_trials=10,trial_len=10,return_states=False):
  exp_len = num_trials*(trial_len+task.pm_maps)
  score = -np.ones([neps,exp_len])
  states = -np.ones([neps,2,exp_len,net.stsize]) # hstate,cstate
  for ep in range(neps):
    # forward prop
    I,S,atarget = task.gen_ep_data(
              num_trials=num_trials,
              trial_len=trial_len,
              pm_probe_positions=[3,7])
    ahat_ulog = net(I,S)
    ep_states = np.stack([net.hstates,net.cstates])
    # print(ep_states.shape)
    states[ep] = ep_states
    # print(net.hstates.shape)
    trial_score = (maxsoftmax(ahat_ulog) == atarget).numpy()
    score[ep] = trial_score.squeeze()
  ev_acc = score.mean(0)
  if not return_states:
    return ev_acc
  else:
    return ev_acc,states

def eval_net(net,task=None,neps=500,gen_data_fn=None,return_states=False):
  if type(gen_data_fn)==type(None):
    gen_data_fn = lambda : task.gen_ep_data(
                    num_trials=4,
                    trial_len=10,
                    pm_probe_positions=[3,7])
  i,s,a = gen_data_fn()
  exp_len = len(i)
  score = -np.ones([neps,exp_len])
  states = -np.ones([neps,2,exp_len,net.stsize]) # hstate,cstate
  for ep in range(neps):
    # forward prop
    I,S,atarget = gen_data_fn()
    ahat_ulog = net(I,S,store_states=True)
    ep_states = np.stack([net.hstates,net.cstates])
    states[ep] = ep_states
    trial_score = (maxsoftmax(ahat_ulog) == atarget).numpy()
    score[ep] = trial_score.squeeze()
  ev_acc = score.mean(0)
  if not return_states:
    return ev_acc
  else:
    return ev_acc,states


def eval_byprobe(net,task,neps=500,num_trials=1,trial_len=20):
  """ 
  returns score array [neps,3], 
  where second dim is probe type: neg,pos,pm
  """
  score = -np.ones([3,neps]) # neg,pos,pm
  for ep in range(neps):
    # eval
    I,S,atarget = task.gen_ep_data(
              num_trials=num_trials,
              trial_len=trial_len,
              pm_probe_positions=[3,7])
    ahat_ulog = net(I,S)
    # break down score
    full_trial_score = (maxsoftmax(ahat_ulog) == atarget).numpy()
    offset = 10
    score[0,ep] = full_trial_score[offset:][(atarget[offset:] == 0).numpy().astype(bool)].mean()
    score[1,ep] = full_trial_score[offset:][(atarget[offset:] == 1).numpy().astype(bool)].mean()
    # print(full_trial_score[offset:][(atarget[offset:] == 1).numpy().astype(bool)].mean())
    score[2,ep] = full_trial_score[(task.pm_maps+3,task.pm_maps+7),0].mean()
  return score

def load_net_and_task(emsetting,num_back,pm_maps,seed):
  net = NetDualPM(emsetting=emsetting,seed=seed)
  task = TaskDualPM(num_back=num_back,pm_maps=pm_maps,seed=seed)
  fname = "LSTM_25-EM_%i-numback_%i-pmmaps_%i-trlen_15-ntrials_2-seed_%i-model.pt"%(
    emsetting,num_back,pm_maps,seed)
  exp_dir = "model_data/DualPM_sweep1-fixstimset/"
  fpath = exp_dir+fname
  net.load_state_dict(tr.load(fpath))
  return net,task



def mov_avg(arr,wind):
  MA = -np.ones(len(arr)-wind)
  for t in range(len(arr)-wind):
    MA[t] = arr[t:t+wind].mean()
  return MA


