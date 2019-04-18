import sys
import torch as tr
import numpy as np

from PMmodel import *

## PARAMS
seed = int(sys.argv[1])
## task
nback=2
ntokens_og=3
num_pmtrials=3
edim_og=5
edim_pm=5
focal=int(sys.argv[2])
pm_weight = int(sys.argv[3])
trseqlen = 25

## network
arch = 'purewm'
stsize = 40
indim = edim_og+edim_pm
batch=1
outdim=3

## training
thresh = .99
nepochs = 1000000


# model fpath
fpath = 'model_data/%s_%i-focal_%i-pmtrials_%i-pmweight_%s-trseqlen_%i-seed_%i'%(
          arch,stsize,focal,num_pmtrials,pm_weight,trseqlen,seed)
print(fpath)

# model and task
if arch=='purewm': net = Net(indim,stsize,outdim)
elif arch=='wmem': net = Net_wmem(indim,stsize,outdim)
task = NBackPMTask(nback,ntokens_og,num_pmtrials,edim_og,edim_pm,focal,seed)

# specify loss and optimizer
loss_weight = tr.FloatTensor([1,1,pm_weight]) 
lossop = tr.nn.CrossEntropyLoss()
optiop = tr.optim.Adam(net.parameters(), lr=0.005)

## eval fun
def eval_(net,task):
  teseqlen = 15
  neps = 1500
  score = -np.ones([neps,teseqlen])
  for ep in range(neps):
    # embedding matrix
    task.sample_emat()
    # generate data
    x_seq,y_seq = task.gen_seq(teseqlen,pm_trial_position=[5,9])
    x_embeds,ytarget = task.embed_seq(x_seq,y_seq)
    # forward prop
    yhat = net(x_embeds)
    ep_score = (ytarget == tr.softmax(yhat,-1).argmax(-1)).float().squeeze()
    score[ep] = ep_score 
  return score

# ### train
print('train')

acc = 0
nembeds = 0
for ep in range(nepochs):
  if ep%(nepochs/10)==0:
    print(ep/nepochs)
    score = eval_(net,task)
    np.save(fpath+'-trep_%i'%ep,score)
  # randomize emat
  if acc>thresh:
    task.sample_emat()
    nembeds+=1
  # generate data
  x_seq,y_seq = task.gen_seq(trseqlen)
  x_embeds,ytarget = task.embed_seq(x_seq,y_seq)
  # forward prop
  yhat = net(x_embeds)
  # collect loss through time
  loss,acc = 0,0
  for yh,yt in zip(yhat,ytarget):
    loss += loss_weight[yh]*lossop(yh,yt)
    acc += yt==tr.argmax(tr.softmax(yh,1))
  acc = acc.numpy()/len(yhat)
  # bp and update
  optiop.zero_grad()
  loss.backward()
  optiop.step()

