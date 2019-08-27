import torch as tr
import numpy as np


tr_uniform = lambda a,b,shape: tr.FloatTensor(*shape).uniform_(a,b)
tr_randn = lambda shape: tr.randn(*shape)

tr_embed_og = lambda shape: tr_uniform(-1,0,shape)
tr_embed_pm = lambda shape: tr_uniform(0,1,shape)
tr_noise_pm = tr_embed_og
tr_noise_og = tr_embed_pm

class TaskPM():

  def __init__(self,num_og_tokens=60,num_back=1,num_pms=3):
    """ 
    """
    self.num_og_tokens=num_og_tokens
    self.num_back=num_back
    self.num_pms=num_pms
    return None

  def gen_ep_data(self,num_trials=1,trial_len=20,add_positive_og_probes=10,pm_probes_per_trial=5):
    """
    first implementation instruction phase is trivially pm_stim_idx=pm_action
    eventually I need someway of remapping pm_maps
    to remap pm_maps, I'll reshuffle the pm_emat
    """
    for trial in range(num_trials):
      resp_stim_seq,resp_action_seq = self.gen_trial_resp_phase(trial_len,add_positive_og_probes,pm_probes_per_trial)
      inst_stim_seq,inst_action_seq = self.gen_trial_inst_phase()

  def gen_trial_inst_phase(self):
    """
    currently throwing away first two stim
    stims 2-9 are pm stim
    """
    pm_action_flags = np.arange(2,2+self.num_pms)
    return pm_action_flags,pm_action_flags

  def embed(self):
    """ 
    currently throwing away first two stim
    stims 2-9 are pm stim
    stims 10+ are og stim
    """
    return None

  def gen_trial_resp_phase(self,trial_len=20,add_positive_og_probes=10,pm_probes_per_trial=5):
    """ 
    still need to implement instruction phase

    """
    # sample OG stimulus sequence
    stim_seq = np.random.randint(10,num_og_tokens,trial_len)
    # include positive OG trials
    positive_og_probes = np.random.randint(0,trial_len,add_positive_og_probes)
    for i in positive_og_probes:
      stim_seq[i] = stim_seq[i-self.num_back]
    # analyze stim_seq to produce action_seq
    action_seq = (stim_seq == np.roll(stim_seq,self.num_back)).astype(int)
    action_seq[:self.num_back] = 0
    # include pm trials
    pm_probe_positions = np.random.randint(0,trial_len,pm_probes_per_trial)
    for pm_probe_pos in sorted(pm_probe_positions):
      # action
      pm_action = np.random.choice(range(2,2+self.num_pms))
      action_seq[pm_probe_pos] = pm_action
      if pm_probe_pos+self.num_back<trial_len:
        action_seq[pm_probe_pos+self.num_back] = 0
      # stim
      pm_stim_idx = pm_action
      stim_seq[pm_probe_pos] = pm_stim_idx
    return stim_seq,action_seq


class PurePM():
  """ proactive interference / arbitrary binding task
  """
  def __init__(self,ntokens=2,stimdim=2,seed=99):
    np.random.seed(seed)
    tr.manual_seed(seed)
    self.ntokens = ntokens
    self.stimdim = stimdim
    # initialize emat
    self.sample_emat()

  def sample_emat(self):
    self.emat = np.random.uniform(0,1,[self.ntokens,self.stimdim])
    self.emat = tr.Tensor(self.emat)
    return None

  def resort_emat(self,resort_mode='permute'):
    if resort_mode == 'permute':
      self.emat = self.emat[np.random.permutation(np.arange(self.ntokens))]
    elif resort_mode == 'roll':
      self.emat = self.emat[np.roll(np.arange(self.ntokens),1)]
    return None

  def gen_ep_data(self,ntrials=2,seqlen=2,switchmaps=False):
    """ 
    seqlen: response_probes_per_trials
      NB length of given trial will be ntokens+seqlen
    output compatible with model input
    instruct_seq: [time,1]
    stim_seq: [time,1,stimdim]
    yseq: [time,1]
    """
    # generate encoding and response sub-sequences for each trial
    encode_instructs_2d = np.array([
        np.random.permutation(np.arange(self.ntokens)) 
        for i in range(ntrials)
    ])
    instruct_seq = np.concatenate([
                        encode_instructs_2d,
                        self.ntokens*np.ones([ntrials,seqlen])],
                      1).astype(int) # (ntrials,seqlen+)
    instruct_seq = instruct_seq.reshape(-1)
    # use above sub-sequences to make x_embedding sequences
    response_probes_int_2d = np.random.randint(0,self.ntokens,[ntrials,seqlen])
    stim_seq_int_2d = np.concatenate([encode_instructs_2d,response_probes_int_2d],1)
    ## flatten 2D array [ntrials,nprobes]->[ntrials*nprobes] and embed stim_seq [ntrials*nprobes,edim]
    stim_seq = []
    # print(stim_seq.shape)
    for trialn,stim_seq_int in enumerate(stim_seq_int_2d): # loop over trials
      if switchmaps:
        self.resort_emat()
      # sequence of stim for current trial
      stim_seq_trial = self.emat[stim_seq_int]
      # sequence of stim 
      stim_seq.extend(stim_seq_trial)
    # format output
    instruct_seq = tr.unsqueeze(tr.LongTensor(instruct_seq),1)
    yseq = tr.unsqueeze(tr.LongTensor(stim_seq_int_2d.reshape(-1)),1)
    stim_seq = tr.stack(stim_seq,0).unsqueeze(1)
    return instruct_seq,stim_seq,yseq



class PMTask_PI():

  def __init__(self,nback=1,ntokens_pm=2,ntokens_og=3,stimdim=2,seed=99):
    """ 
    multiple trials
    pm tokens presented at beginning of trial
      order of pm token determines appropriate response
    presentation of pm cue ends trial
    """
    np.random.seed(seed)
    tr.manual_seed(seed)
    self.nback = nback
    # embedding
    self.ntokens_pm = ntokens_pm
    self.ntokens_og = ntokens_og
    self.stimdim = stimdim
    # emat
    self.randomize_emat()
    return None

  def gen_xseq_singletrial(self,min_trial_len=2,max_trial_len=3):
    """
    generates a single trial xseq 
    OG tokens: 0...ntokens_og 
    PM tokens: ntokens_og...ntokens_og+ntokens_pm
    """
    # random length OG task seq
    trial_len = np.random.randint(min_trial_len,max_trial_len+1)
    task_seq = np.random.randint(0,self.ntokens_og,trial_len)
    # pm stim
    pm_cue_encode_seq = np.arange(self.ntokens_og,self.ntokens_pm+self.ntokens_og)
    np.random.shuffle(pm_cue_encode_seq)
    trial_pm_stim = np.random.choice(pm_cue_encode_seq)
    # concat
    seq = np.concatenate([
            pm_cue_encode_seq,
            task_seq,
            [trial_pm_stim]
          ])
    return seq

  def xseq2yseq_singletrial(self,xseq):
    """
    analyzes xseq [1d np]
    returns yseq [1d np]
    in Yseq: 
      0,1 used for og task
      2,... ntokens_pm used for pm probes
    """
    # sequence for ongoing task
    og_xseq = xseq[self.ntokens_pm:-1]
    og_xseq_roll = np.roll(og_xseq,self.nback)
    og_yseq = np.concatenate([
      np.zeros(self.nback),
      (og_xseq[self.nback:] == og_xseq_roll[self.nback:]).astype(int)])
    # determine final pm response
    pm_responses_encode = np.arange(2,2+self.ntokens_pm)
    pm_response_final = list(xseq).index(xseq[-1])+2
    # concatenate
    yseq = np.concatenate([
      pm_responses_encode,
      og_yseq,
      [pm_response_final]
      ])
    return yseq

  def xseq2tseq_singletrial(self,xseq):
    tseq = np.concatenate([
            np.arange(1,self.ntokens_pm+1),
            np.zeros(len(xseq)-self.ntokens_pm)
            ])
    return tseq

  def gen_seqs_multitrial(self,min_trial_len=2,max_trial_len=3,ntrials=2):
    """ 
    generates multiple trial sequences (tseq,xseq,yseq)
    random length
    pm ends trial
    returns seq `1D`[np]
    """
    xseq = np.array([])
    yseq = np.array([])
    tseq = np.array([])
    for trial in range(ntrials):
      xseq_ = self.gen_xseq_singletrial(min_trial_len,max_trial_len)
      yseq_ = self.xseq2yseq_singletrial(xseq_)
      tseq_ = self.xseq2tseq_singletrial(xseq_)
      xseq = np.concatenate([xseq,xseq_])
      yseq = np.concatenate([yseq,yseq_])
      tseq = np.concatenate([tseq,tseq_])
    return tseq,xseq,yseq

  def randomize_emat(self):
    self.emat = tr_uniform(0,1,[self.ntokens_pm+self.ntokens_og,self.stimdim])
    return None

  def embed_xseq(self,xseq):
    """ 
    takes 1d seq
    returns embedded seq
    output should be compatible 
      with input to model
    """
    xseq_embed = self.emat[xseq]
    return xseq_embed

  def gen_ep_data(self,min_trial_len=2,max_trial_len=3,ntrials=2):
    """ top wrapper
    randomly generates an episode 
    input is [np] output is [tr]
    output compatible with model input
    """
    # self.randomize_emat()
    tseq,xseq,yseq = self.gen_seqs_multitrial(min_trial_len,max_trial_len,ntrials)
    xseq_embed = self.embed_xseq(xseq)
    # np to torch
    tseq = tr.unsqueeze(tr.LongTensor(tseq),1)
    xseq_embed = tr.unsqueeze(tr.Tensor(xseq_embed),1)
    yseq = tr.unsqueeze(tr.LongTensor(yseq),1)
    return tseq,xseq_embed,yseq



class PMTask_Focality():
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
    pm_noise = tr_noise_pm([len(pm_embeds),self.pm_noise_dim])
    og_noise = tr_noise_og([len(og_embeds),self.og_noise_dim])
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
    self.emat_og = tr_embed_og([self.ntokens_og,self.og_signal_dim])
    self.emat_pm = tr_embed_pm([self.ntokens_pm,self.pm_signal_dim])

