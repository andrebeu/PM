""" first sweep on dual PM task
"""

import sys

import torch as tr
import numpy as np

from PM_models import *
from PM_tasks import *
from helperfuns_dualPM import *


# params
num_back = int(sys.argv[1])
pm_maps = int(sys.argv[2])
seed = int(sys.argv[3])
EMsetting = 2

# initialize
task = TaskDualPM(num_back=num_back,pm_maps=pm_maps,seed=seed)
net = NetDualPM(emsetting=EMsetting,seed=seed)

# paths
sim_dir = 'model_data/DualPM_sweep1-fixstimset/'
model_fname = "LSTM_25-EM_%i-numback_%i-pmmaps_%i-trlen_15-ntrials_2-seed_%i"%(
                EMsetting,num_back,pm_maps,seed)
fpath = sim_dir+model_fname

## train
tracc = train_net(net,task,neps=50000)
# save
np.save(fpath+'-tracc',tracc)
tr.save(net.state_dict(),fpath+'-model.pt')

