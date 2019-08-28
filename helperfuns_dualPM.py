from PM_models import *
from PM_tasks import *
import torch as tr
import numpy as np

from sklearn.metrics import pairwise_distances


maxsoftmax = lambda ulog: tr.argmax(tr.softmax(ulog,-1),-1)


def train_net(net,task,neps,gen_data_fn=None):
  if type(gen_data_fn)==type(None):
    gen_data_fn = lambda : task.gen_ep_data(
                    num_trials=2,
                    trial_len=10,
                    pos_og_bias=np.random.randint(1,100,1),
                    pm_probes_per_trial=np.random.randint(0,10,1))
  lossop = tr.nn.CrossEntropyLoss()
  optiop = tr.optim.Adam(net.parameters(), lr=0.001)
  acc = -np.ones(neps)
  task.randomize_emat()
  for ep in range(neps):
    # generate data
    # pos_og_bias = 
    # pm_probes_per_trial = 
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
  return acc

def eval_net(net,task,neps_ev=500,num_trials=2,trial_len=15,og_bias=[1,100]):
  exp_len = num_trials*(trial_len+task.num_pm_maps)
  score = -np.ones([neps_ev,exp_len])
  for ep in range(neps_ev):
    pos_og_bias = np.random.randint(og_bias[0],og_bias[1],1)
    # forward prop
    I,S,atarget = task.gen_ep_data(
              num_trials=num_trials,
              trial_len=trial_len,
              pos_og_bias=pos_og_bias,
              pm_probe_positions_=[3,7])
    ahat_ulog = net(I,S)
    trial_score = (maxsoftmax(ahat_ulog) == atarget).numpy()
    score[ep] = trial_score.squeeze()
  ev_acc = score.mean(0)
  return ev_acc

def eval_byprobe(net,task,neps_ev=500,num_trials=1,trial_len=20):
  """ 
  returns score array [neps,3], 
  where second dim is probe type: neg,pos,pm
  """
  score = -np.ones([3,neps_ev]) # neg,pos,pm
  for ep in range(neps_ev):
    # eval
    I,S,atarget = task.gen_ep_data(
              num_trials=num_trials,
              trial_len=trial_len,
              pos_og_bias=50,
              pm_probe_positions_=[3,7])
    ahat_ulog = net(I,S)
    # break down score
    full_trial_score = (maxsoftmax(ahat_ulog) == atarget).numpy()
    offset = 10
    score[0,ep] = full_trial_score[offset:][(atarget[offset:] == 0).numpy().astype(bool)].mean()
    score[1,ep] = full_trial_score[offset:][(atarget[offset:] == 1).numpy().astype(bool)].mean()
    score[2,ep] = full_trial_score[(task.num_pm_maps+3,task.num_pm_maps+7),0].mean()
  return score

def load_net_and_task(num_back,num_pm_maps,seed):
  task = TaskDualPM(num_back=num_back,num_pm_maps=num_pm_maps,seed=seed)
  net = NetDualPM(seed=seed)
  fname = "LSTM_25-EM_0-numback_%i-pmmaps_%i-trlen_15-ntrials_2-seed_%i-model.pt"%(
    num_back,num_pm_maps,seed)
  exp_dir = "model_data/DualPM_sweep1-fixstimset/"
  fpath = exp_dir+fname
  net.load_state_dict(tr.load(fpath))
  return net,task



def mov_avg(arr,wind):
  MA = -np.ones(len(arr)-wind)
  for t in range(len(arr)-wind):
    MA[t] = arr[t:t+wind].mean()
  return MA


