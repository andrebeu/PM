import sys
import torch as tr
import numpy as np

from PMmodel import *

## PARAMS

# network
seed = int(sys.argv[1])
arch = str(sys.argv[2])
stsize = int(sys.argv[3])
edim = 5
batch = 1
outdim = 3
# task
nback=2
num_og_tokens = 3
num_pm_trials = int(sys.argv[4])
# training
train_tresh = .99
nepochs = 100000
train_seqlen = 25
pm_loss_weight = int(sys.argv[5])

fpath = 'model_data/%s_%i-pmtrials_%i-trseqlen_%i-pmweight_%s-seed_%i'%(
          arch,stsize,num_pm_trials,train_seqlen,pm_loss_weight,seed)
print(fpath)


## initialize model and task
if arch=='purewm':
  net = Net(edim,stsize,outdim,seed)
elif arch=='wmem':
  net = Net_wmem(edim,stsize,outdim,seed)
task = NBackPMTask(nback,num_og_tokens,num_pm_trials,seed)


## train
print('training')

# specify loss and optimizer
lossop = tr.nn.CrossEntropyLoss(weight=tr.Tensor([1,1,pm_loss_weight]))
optiop = tr.optim.Adam(net.parameters(), lr=0.005)

# init arrays
L = -np.ones([nepochs])
A = -np.ones([nepochs])
E = -np.ones([nepochs])

acc = 0
nembeds = 0
Emat = tr.FloatTensor(num_og_tokens+1,edim).uniform_(0,1)
for ep in range(nepochs):
  if ep%(nepochs/5)==0:
    print(ep/nepochs,nembeds)
  optiop.zero_grad() 
  # randomize emat
  if acc>train_tresh:
    Emat = tr.FloatTensor(num_og_tokens+1,edim).uniform_(0,1)
    nembeds+=1
  # generate data
  x_int,ytarget = task.gen_seq(train_seqlen)
  ytarget = tr.LongTensor(ytarget).unsqueeze(1)
  x_embeds = Emat[x_int].unsqueeze(1) 
  # forward prop
  yhat = net(x_embeds)
  # collect loss through time
  loss,acc = 0,0
  for yh,yt in zip(yhat,ytarget):
    loss += lossop(yh,yt)
    acc += yt==tr.argmax(tr.softmax(yh,1))
  acc = acc.cpu().numpy()/train_seqlen
  # bp and update
  loss.backward()
  optiop.step()
  epoch_loss = loss.item()
  L[ep] = epoch_loss
  A[ep] = acc
  E[ep] = nembeds


## eval
print('eval')

seqlen = 15
neps = 1500
score = -np.ones([neps,seqlen])

for ep in range(neps):
  # embedding matrix
  Emat = tr.FloatTensor(num_og_tokens+1,edim).uniform_(0,1)
  # generate data
  x_int,ytarget = task.gen_seq(seqlen,pm_trial_position=[5,9])
  ytarget = tr.LongTensor(ytarget).unsqueeze(1)
  # embed inputs
  x_embeds = Emat[x_int]
  x_embeds = x_embeds.unsqueeze(1)
  # forward prop
  yhat = net(x_embeds)
  ep_score = (ytarget == tr.softmax(yhat,-1).argmax(-1)).float().squeeze()
  score[ep] = ep_score 

## save
np.save(fpath,score)
