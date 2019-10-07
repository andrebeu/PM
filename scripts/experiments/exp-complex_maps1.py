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
num_back = 2
stsize = 20
pm_maps = 3 
seed = int(sys.argv[1])
EMsetting = 1

## initialize
task = TaskDualPM(num_back=num_back,pm_maps=pm_maps,seed=seed)
net = NetDualPM(stsize=stsize,emsetting=EMsetting,seed=seed)


fpath = 'model_data/complex_maps_sweep1/'
fpath += "LSTM%i-EM1-nback_%i-pm_maps_%i-seed_%i"%(stsize,num_back,pm_maps,seed)


## similar to a complex span task
gen_data_fn = lambda : task.gen_ep_data(num_trials=2,trial_len=20,pm_probe_positions=np.arange(10,20))
tracc = train_net(net,task,neps=150000,gen_data_fn=gen_data_fn,verb=True)

np.save(fpath+'-tracc',tracc)
tr.save(net.state_dict(),fpath+'-model.pt')
