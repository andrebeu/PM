import torch as tr
import numpy as np

tr_uniform = lambda shape: tr.FloatTensor(*shape).uniform_(0,1)
tr_randn = lambda shape: tr.randn(*shape)

tr_noise = tr_uniform 
tr_embed = tr_uniform


class NBackPMTask():

  def __init__(self,nback,num_pm_trials,
    og_signal_dim,pm_signal_dim,og_noise_dim,pm_noise_dim,
    ntokens_og,seed):
    """ 
    """
    np.random.seed(seed)
    tr.manual_seed(seed)
    self.nback = nback
    self.ntokens_pm = 1
    self.ntokens_og = ntokens_og
    # trials
    self.pm_token = ntokens_og
    self.num_pm_trials = num_pm_trials
    self.min_start_trials = 1
    # focality
    self.og_signal_dim = og_signal_dim
    self.pm_signal_dim = pm_signal_dim
    self.og_noise_dim = og_noise_dim
    self.pm_noise_dim = pm_noise_dim
    assert og_signal_dim+og_noise_dim==pm_signal_dim+pm_noise_dim
    self.sample_emat()
    return None

  def gen_seq(self,ntrials=20,pm_trial_position=None):
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
    X_embed = -tr.ones(len(X_seq),self.og_signal_dim+self.og_noise_dim)
    # find trials of corresponding types
    pm_trials_bool = X_seq >= self.ntokens_og
    pm_trials = np.where(pm_trials_bool)
    og_trials = np.where(np.logical_not(pm_trials_bool))
    # take signal_dim (time,edim_signal_dim)
    pm_embeds = self.emat_pm[X_seq[pm_trials] - self.ntokens_og] 
    og_embeds = self.emat_og[X_seq[og_trials]] 
    # make noise (time,edim_noise)
    pm_noise = tr_noise([len(pm_embeds),self.pm_noise_dim])
    og_noise = tr_noise([len(og_embeds),self.og_noise_dim])
    # cat signal_dim and noise (time,edim)
    pm_embeds = tr.cat([pm_embeds,pm_noise],-1)
    og_embeds = tr.cat([og_noise,og_embeds],-1)
    # put into respective positions
    X_embed[pm_trials] = pm_embeds
    X_embed[og_trials] = og_embeds 
    # include batch dim   
    X_embed = tr.unsqueeze(X_embed,1)
    Y_embed = tr.unsqueeze(tr.LongTensor(Y_seq),1)
    return X_embed,Y_embed

  def sample_emat(self):
    self.emat_og = tr_embed([self.ntokens_og,self.og_signal_dim])
    self.emat_pm = tr_embed([self.ntokens_pm,self.pm_signal_dim])



class PMNet(tr.nn.Module):

  def __init__(self,indim,stsize,outdim,EM=True,seed=132):
    super().__init__()
    # seed
    tr.manual_seed(seed)
    # params
    self.indim = indim
    self.stsize = stsize
    self.outdim = outdim
    # layers
    self.ff_in = tr.nn.Linear(indim,stsize)
    self.relu_in = tr.nn.ReLU()
    self.initial_state = tr.rand(2,1,self.stsize,requires_grad=True)
    self.cell = tr.nn.LSTMCell(stsize,stsize)
    self.ff_out1 = tr.nn.Linear(stsize,stsize)
    self.relu_out = tr.nn.ReLU()
    self.ff_out2 = tr.nn.Linear(stsize,outdim)
    # retrival layers
    self.rgate = tr.nn.Linear(stsize,stsize)
    self.sigm = tr.nn.Tanh()
    self.EM = EM
    return None

  def forward(self,xdata):
    """ 
    input: xdata `(time,batch,embedding)`
    output: yhat `(time,batch,outdim)`
    """
    seqlen = xdata.shape[0]
    # inproj
    percept = self.ff_in(xdata)
    percept = self.relu_in(percept)
    # compute retrieval similarities
    percept_pm_cue = percept[0]
    sim = (tr.cosine_similarity(percept_pm_cue,percept,dim=-1) + 1).detach()/2
    self.sim = sim
    # unroll
    lstm_outs = -tr.ones(seqlen,1,self.stsize)
    ## PM cue encoding trial
    # compute internal state for pm_cue 
    lstm_output,lstm_state = self.initial_state
    lstm_output_pm,lstm_state_pm = self.cell(percept_pm_cue,(lstm_output,lstm_state))
    lstm_outs[0] = lstm_output_pm
    ## task
    lstm_output,lstm_state = lstm_output_pm,lstm_state_pm
    self.rgate_act = -tr.ones(seqlen,self.stsize)
    for t in range(1,seqlen):
      if self.EM:
        # rgate act
        rgate_act = self.rgate(percept[t,:,:])
        rgate_act = self.sigm(rgate_act)
        self.rgate_act[t] = rgate_act
        # update state based on similarity to pm_state memory
        # lstm_state = sim[t,0]*lstm_state_pm + (1-sim[t,0])*lstm_state
        lstm_state = rgate_act*lstm_state_pm + lstm_state
      # compute cell prediction
      lstm_output,lstm_state = self.cell(percept[t,:,:],(lstm_output,lstm_state))
      lstm_outs[t] = lstm_output
    # outporj
    yhat = self.ff_out1(lstm_outs)
    yhat = self.relu_out(yhat)
    yhat = self.ff_out2(yhat)
    return yhat

