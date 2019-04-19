import sys
import torch as tr
import numpy as np

from PMmodel import *

## sweeping params
seed = int(sys.argv[1])
stsize = int(sys.argv[2]) # 30
num_pmtrials = int(sys.argv[3]) # 5
pm_weight = int(sys.argv[4]) # 1
EM=int(sys.argv[5]) 
nback=int(sys.argv[6]) # 1

## constant params

# task
ntokens_og=3
edim_og=8
edim_pm=0
focal=1
trseqlen = 20

# network
indim = edim_og+edim_pm
batch=1
outdim=3

## training
thresh = .95
nepochs = 100000

tr.manual_seed(seed)
np.random.seed(seed)

# model fpath
fpath = 'model_data/EM_%i-stsize_%i-nback_%i-focal_%i-pmtrials_%i-pmweight_%s-seed_%i'%(
          EM, stsize, nback, focal, num_pmtrials, pm_weight, seed)
print(fpath)

# model and task
net = PMNet(indim,stsize,outdim,EM,seed)
task = NBackPMTask(nback,ntokens_og,num_pmtrials,edim_og,edim_pm,focal,seed)


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

# specify loss and optimizer
loss_weight = tr.FloatTensor([1,1,pm_weight]) 
lossop = tr.nn.CrossEntropyLoss()
optiop = tr.optim.Adam(net.parameters(), lr=0.0005)

acc = 0
nembeds = 0
for ep in range(nepochs):
  if ep%(nepochs/10)==0:
    print(ep/nepochs,nembeds)
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
    loss += loss_weight[yt]*lossop(yh,yt)
    acc += yt==tr.argmax(tr.softmax(yh,1))
  acc = acc.numpy()/len(yhat)
  # bp and update
  optiop.zero_grad()
  loss.backward()
  optiop.step()

score = eval_(net,task)
np.save(fpath+'-trep_final',score)