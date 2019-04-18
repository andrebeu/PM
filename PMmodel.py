import torch as tr
import numpy as np

tr_uniform = lambda shape: tr.FloatTensor(*shape).uniform_(0,1)

class NBackPMTask():

  def __init__(self,nback,ntokens_og,num_pm_trials,edim_og,edim_pm,focal,seed=132):
    """ 
    """
    np.random.seed(seed)
    self.nback = nback
    self.ntokens_og = ntokens_og
    self.ntokens_pm = 1
    self.pm_token = ntokens_og
    self.min_start_trials = 1
    self.num_pm_trials = num_pm_trials
    # embedding
    self.emat_initialized=True
    self.edim_og = edim_og
    self.edim_pm = edim_pm
    self.focal = focal
    self.sample_emat()
    return None

  def gen_seq(self,ntrials=30,pm_trial_position=None):
    """
    if pm_trial_position is not specified, they are randomly sampled
      rand pm_trial_position for training, fixed for eval
    """
    # insert ranomly positioned pm trials
    if type(pm_trial_position)==type(None):
      ntrials -= 1+self.num_pm_trials
      pm_trial_position = np.random.randint(self.min_start_trials,ntrials,self.num_pm_trials) 
    else:
      ntrials -= 1+len(pm_trial_position)
      pm_trial_position = pm_trial_position
    # generate og stim
    seq = np.random.randint(0,self.ntokens_og,ntrials)
    X = np.insert(seq,[0,*pm_trial_position],self.pm_token)
    # form Y 
    Xroll = np.roll(X,self.nback)
    Y = (X == Xroll).astype(int) # nback trials
    Y[X==self.pm_token]=2 # pm trials
    return X,Y

  def embed_seq(self,X_seq,Y_seq):
    """ 
    takes 1-D input sequences
    returns 
      X_embed `(time,batch,edim)`[torch]
      Y_embed `(time,batch)`[torch]
    """
    pm_trials_bool = X_seq >= self.ntokens_og
    pm_trials = np.where(pm_trials_bool)
    og_trials = np.where(np.logical_not(pm_trials_bool))
    if self.focal:
      X_embed = self.emat[X_seq]
    else:
      X_embed = -tr.ones(len(X_seq),self.edim_og+self.edim_pm)
      # pm trials
      pm_embeds = self.emat_pm[X_seq[pm_trials] - self.ntokens_og] # (time,edim)
      pm_noise = tr_uniform([len(pm_embeds),self.edim_og])
      pm_embeds = tr.cat([pm_embeds,pm_noise],1)
      X_embed[pm_trials] = pm_embeds
      # og trials
      og_embeds = self.emat_og[X_seq[og_trials]] # (time,edim)
      og_noise = tr_uniform([len(og_embeds),self.edim_pm])
      og_embeds = tr.cat([og_noise,og_embeds],1)
      X_embed[og_trials] = og_embeds 
    # include batch dim   
    X_embed = tr.unsqueeze(X_embed,1)
    Y_embed = tr.unsqueeze(tr.LongTensor(Y_seq),1)
    return X_embed,Y_embed

  def sample_emat(self):
    if self.focal:
      self.emat = tr_uniform([self.ntokens_og+self.ntokens_pm,self.edim_og+self.edim_pm])
    else:
      self.emat_og = tr_uniform([self.ntokens_og,self.edim_og])
      self.emat_pm = tr_uniform([self.ntokens_pm,self.edim_pm])



class Net(tr.nn.Module):
  def __init__(self,indim,stsize,outdim,seed=132):
    super().__init__()
    # seed
    tr.manual_seed(seed)
    # params
    self.indim = indim
    self.stsize = stsize
    self.outdim = outdim
    # layers
    self.ff_in = lambda x: tr.nn.ReLU()(tr.nn.Linear(indim,stsize)(x))
    self.initial_state = tr.rand(2,1,self.stsize,requires_grad=True)
    self.cell = tr.nn.LSTMCell(stsize,stsize)
    self.ff_out = tr.nn.Linear(stsize,outdim)
    return None

  def forward(self,xdata):
    """ 
    input: xdata `(time,batch,embedding)`
    output: yhat `(time,batch,outdim)`
    """
    seqlen = xdata.shape[0]
    # init cell state
    lstm_output,lstm_state = self.initial_state
    # inproj
    xdata = self.ff_in(xdata)
    # unroll
    lstm_outs = -tr.ones(seqlen,1,self.stsize)
    for t in range(seqlen):
      lstm_output,lstm_state = self.cell(xdata[t,:,:],(lstm_output,lstm_state))
      lstm_outs[t] = lstm_output
    # outporj
    yhat = self.ff_out(lstm_outs)
    # yhat = tr.nn.Softmax(dim=-1)(yhat)
    return yhat


class Net_wmem(tr.nn.Module):
  def __init__(self,indim=2,stsize=4,outdim=3,seed=132):
    super().__init__()
    # seed
    tr.manual_seed(seed)
    # params
    self.indim = indim
    self.stsize = stsize
    self.outdim = outdim
    # layers
    self.ff_in = lambda x: tr.nn.ReLU()(tr.nn.Linear(indim,stsize)(x))
    self.initial_state = tr.rand(2,1,self.stsize,requires_grad=True)
    self.cell = tr.nn.LSTMCell(stsize,stsize)
    self.ff_out = tr.nn.Linear(stsize,outdim)
    return None

  def forward(self,xdata):
    """ 
    input: xdata `(time,batch,embedding)`
    output: yhat `(time,batch,outdim)`
    """
    seqlen = xdata.shape[0]
    # inproj
    percept = self.ff_in(xdata)
    # compute retrieval similarities
    percept_pm_cue = percept[0]
    sim = self.sim = (tr.cosine_similarity(percept_pm_cue,percept,dim=-1) + 1).detach()/2
    # unroll
    lstm_outs = -tr.ones(seqlen,1,self.stsize)
    ## PM cue encoding trial
    # compute internal state for pm_cue 
    lstm_output,lstm_state = self.initial_state
    lstm_output_pm,lstm_state_pm = self.cell(percept_pm_cue,(lstm_output,lstm_state))
    lstm_outs[0] = lstm_output_pm
    ## task
    lstm_output,lstm_state = lstm_output_pm,lstm_state_pm
    for t in range(1,seqlen):
      # update state based on similarity to pm_state memory
      lstm_state = sim[t,0]*lstm_state_pm+(1-sim[t,0])*lstm_state
      # compute cell prediction
      lstm_output,lstm_state = self.cell(percept[t,:,:],(lstm_output,lstm_state))
      lstm_outs[t] = lstm_output
    # outporj
    yhat = self.ff_out(lstm_outs)
    return yhat

