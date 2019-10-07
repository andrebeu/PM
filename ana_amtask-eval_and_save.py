
import torch as tr
import numpy as np

import itertools

from PM_models import *
from PM_tasks import *
from help_amtask import *



## train params
em_trL = [0,1]
ntrials_trL = [1,2]
trlen_trL = [20,30,40]
ntoksurpL = [0,50]
## eval params
neps_ev = 400
ntr_ev = 20
trlen_ev = 10

for em_tr,ntrials_tr,trlen_tr,ntoksurp in itertools.product(em_trL,ntrials_trL,trlen_trL,ntoksurpL):
  print(em_tr,ntrials_tr,trlen_tr,ntoksurp)
  netL = load_netL(em_tr,ntrials_tr,trlen_tr,ntoksurp)
  for net in netL:
    eval_and_save(net,neps_ev,ntr_ev,trlen_ev)