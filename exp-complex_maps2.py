""" 
complex maps task: 
3 phases: 
  (i) instruction/encoding phase (as before)
  (ii) OG task performance on k probes
  (iii) maps performance
this is similar to a complex span, 
  where the OG task serves to consume WM
  and where the contents to be held are maps
"""

import sys

import torch as tr
import numpy as np

from PM_models import *
from PM_tasks import *
from helperfuns_dualPM import *


## task sweep params
seed = int(sys.argv[1])
num_back = int(sys.argv[2])
num_maps = int(sys.argv[3]) 
stsize = 20
EMsetting = 1
num_trials = 3
trial_len = 16
start_maps_task = 8

## initialize
task = TaskDualPM(num_back=num_back,pm_maps=num_maps,seed=seed)
net = NetDualPM(stsize=stsize,emsetting=EMsetting,seed=seed)

fpath = 'model_data/complex_maps_sweep2/'
fpath += "LSTM%i-EM1-nback_%i-num_maps_%i-num_trials_%i-trial_len_%i-seed_%i"%(
            stsize,num_back,num_maps,num_trials,trial_len,seed)


## similar to a complex span task
gen_data_fn = lambda : task.gen_ep_data(
                          num_trials=num_trials,
                          trial_len=trial_len,
                          pm_probe_positions=np.arange(start_maps_task,trial_len)
                          )

tracc = train_net(net,task,neps=150000,gen_data_fn=gen_data_fn,verb=True)

np.save(fpath+'-tracc',tracc)
tr.save(net.state_dict(),fpath+'-model.pt')
