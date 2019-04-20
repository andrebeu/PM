import sys
import torch as tr
import numpy as np

from PMmodel import *

## sweeping params
seed = int(sys.argv[1])
signal = int(sys.argv[2])
pmweight = int(sys.argv[3]) 
EM = int(sys.argv[4]) 


### constant
# task
nback=1
pmtrials = 5
ntokens_og = 3
edim = 8
noise = edim-signal
og_signal_dim = signal
pm_signal_dim = signal
og_noise_dim = noise
pm_noise_dim = noise


# network
stsize = 30
indim = og_signal_dim+og_noise_dim
batch=1
outdim=3

# training
nepochs = 100000
thresh = .95
trseqlen=20



###

# model and task
net = PMNet(indim,stsize,outdim,EM)
task = NBackPMTask(nback,pmtrials,og_signal_dim,pm_signal_dim,og_noise_dim,pm_noise_dim,ntokens_og,seed)

tr.manual_seed(seed)
np.random.seed(seed)

# model fpath
fpath = 'model_data/EM_%i-stsize_%i-nback__%i-pmtrials_%i-pmweight_%s-signal_%i-noise_%i-seed_%i'%(
          EM, stsize, nback, pmtrials, pmweight, signal, noise, seed)
print(fpath)

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
loss_weight = tr.FloatTensor([1,1,pmweight]) 
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
np.save(fpath+'-trep_nepochs',score)