import torch as tr
import numpy as np


class NBackPMTask():

  def __init__(self,nback,num_og_tokens,num_pm_trials,seed=132):
    """ 
    """
    np.random.seed(seed)
    self.nback = nback
    self.num_og_tokens = num_og_tokens
    self.pm_token = num_og_tokens
    self.min_start_trials = 1
    self.num_pm_trials = num_pm_trials
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
    seq = np.random.randint(0,self.num_og_tokens,ntrials)
    X = np.insert(seq,[0,*pm_trial_position],self.pm_token)
    # form Y 
    Xroll = np.roll(X,self.nback)
    Y = (X == Xroll).astype(int) # nback trials
    Y[X==self.pm_token]=2 # pm trials
    return X,Y


class Net(tr.nn.Module):
  def __init__(self,edim=2,stsize=4,outdim=3,seed=132):
    super().__init__()
    # seed
    tr.manual_seed(seed)
    # params
    self.edim = edim
    self.stsize = stsize
    self.outdim = outdim
    # layers
    self.ff_in = tr.nn.Linear(edim,stsize)
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
  def __init__(self,edim=2,stsize=4,outdim=3,seed=132):
    super().__init__()
    # seed
    tr.manual_seed(seed)
    # params
    self.edim = edim
    self.stsize = stsize
    self.outdim = outdim
    # layers
    self.ff_in = tr.nn.Linear(edim,stsize)
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

