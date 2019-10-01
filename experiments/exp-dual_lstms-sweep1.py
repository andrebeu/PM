import sys

import torch as tr
import numpy as np

from PM_models import *
from PM_tasks import *
from helperfuns_dualPM import *

seed = int(sys.argv[1])
ntrials = int(sys.argv[2])
trlen = int(30/ntrials)
stsize1 = int(sys.argv[3])
stsize2 = int(sys.argv[4])

nmaps = 8

fpath = "model_data/dual_lstms_sweep1/"
fpath += "lstm1_%i-lstm2_%i-ntrials_%i-seed_%i"%(stsize1,stsize2,ntrials,seed)

def gen_data_fn(task,ntrials,trlen):
  i,x,y = task.gen_ep_data(num_trials=ntrials,
                           trial_len=trlen,
                           pm_probe_positions=np.arange(trlen))
  y = y.reshape(ntrials,trlen+nmaps)[:,nmaps:].reshape(-1,1)
  return i,x,y

neps_train = 200000

net = NetDualLSTMToy(stsize1=stsize1,stsize2=stsize2,seed=seed)
task = TaskDualPM(num_back=1,nmaps=nmaps,seed=seed)
lam_gen_data_fn = lambda: gen_data_fn(task,ntrials,trlen)

tr_acc = train_net(net,task,neps=neps_train,gen_data_fn=lam_gen_data_fn,verb=True)

## saving
np.save(fpath+"-tracc",tr_acc)
tr.save(net.state_dict(),fpath+'-model.pt')