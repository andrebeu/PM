""" first sweep on dual PM task
"""

import sys

import torch as tr
import numpy as np

from PM_models import *
from PM_tasks import *


## params

# task sweep params
num_back = int(sys.argv[1])
num_pm_maps = int(sys.argv[2])
seed = int(sys.argv[3])

## initialize
task = TaskDualPM(num_back=num_back,num_pm_maps=num_pm_maps,seed=seed)
net = NetDualPM(seed=seed)


sim_dir = 'model_data/DualPM_sweep1-fixstimset/'
model_fname = "LSTM_25-EM_0-numback_%i-pmmaps_%i-trlen_15-ntrials_2-seed_%i"%(
                num_back,num_pm_maps,seed)
print(model_fname)


maxsoftmax = lambda ulog: tr.argmax(tr.softmax(ulog,-1),-1)


def train(net,task,neps,num_trials,trial_len):
  lossop = tr.nn.CrossEntropyLoss()
  optiop = tr.optim.Adam(net.parameters(), lr=0.001)
  exp_len = num_trials*(trial_len+task.num_pm_maps)
  acc = -np.ones(neps)
  for ep in range(neps):
    # generate data
    pos_og_bias = np.random.randint(1,trial_len*100,1)
    pm_probes_per_trial = np.random.randint(0,trial_len,1)
    I,S,atarget = task.gen_ep_data(
                    num_trials=num_trials,
                    trial_len=trial_len,
                    pos_og_bias=pos_og_bias,
                    pm_probes_per_trial=pm_probes_per_trial)
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


## train
num_trials_tr = 2
trial_len_tr = 15
neps_tr = 20000

tr_acc = train(net,task,
               neps=neps_tr,
               num_trials=num_trials_tr,
               trial_len=trial_len_tr
              )

fpath = sim_dir+model_fname
tr.save(net.state_dict(),fpath+'-model.pt')

