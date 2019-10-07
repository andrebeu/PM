import sys

import torch as tr
import numpy as np

from PM_models import *
from PM_tasks import *
from helperfuns_dualPM import *


## task sweep params
num_back = 2
pm_maps = 3 
seed = int(sys.argv[1])
EMsetting = 1

## initialize
task = TaskDualPM(num_back=num_back,pm_maps=pm_maps,seed=seed)
net = NetDualPM(emsetting=EMsetting,seed=seed)

fpath = 'model_data/DualPM_sweep-2trials/'
fpath += "LSTM25-EM1-nback_%i-pm_maps_%i-seed_%i"%(num_back,pm_maps,seed)


gen_data_fn = lambda : task.gen_ep_data(
                          num_trials=2,
                          trial_len=20,
                          )

tracc = train_net(net,task,neps=100000,gen_data_fn=gen_data_fn,verb=True)
evacc1 = eval_net(net,task)
evacc2 = eval_byprobe(net,task)
evacc2 = np.nan_to_num(evacc2,0).mean(1)

np.save(fpath+'-tracc',tracc)
np.save(fpath+'-evacc',evacc1)
np.save(fpath+'-evacc_byprobe',evacc2)
tr.save(net.state_dict(),fpath+'-model.pt')
