import torch as tr
import numpy as np

from sklearn.metrics import pairwise_distances


"""
assumes first input is the PM cue
"""

class PINet(tr.nn.Module):

  def __init__(self,stimdim,stsize,outdim,ninstructs,EMbool=True,seed=132):
    super().__init__()
    # seed
    tr.manual_seed(seed)
    # layer sizes
    self.stimdim = stimdim
    self.instdim = stimdim
    self.ninstructs = ninstructs
    self.resp_trial_flag = ninstructs-1
    self.stsize = stsize
    self.outdim = outdim
    # instruction layer
    self.embed_instruct = tr.nn.Embedding(self.ninstructs,self.instdim)
    self.i2inst = tr.nn.Linear(self.instdim,self.instdim,bias=False) 
    # sensory layer
    self.lstm_stim = tr.nn.LSTMCell(self.stimdim,self.stimdim) # x2stim
    self.initst_stim = tr.rand(2,1,self.stimdim,requires_grad=True)
    self.ff_stim = tr.nn.Linear(self.stimdim,self.stimdim,bias=False)
    # Main LSTM CELL
    self.lstm_main = tr.nn.LSTMCell(self.stimdim+self.instdim,self.stsize)
    self.initst_main = tr.rand(2,1,self.stsize,requires_grad=True)
    # out proj
    self.cell2outhid = tr.nn.Linear(self.stsize,self.stsize,bias=False)
    self.ff_hid2ulog = tr.nn.Linear(self.stsize,self.outdim,bias=False)
    # Episodic memory
    self.EMbool = EMbool
    self.EM_key = []
    self.EM_value = []
    # self.ff_em2cell = tr.nn.Linear(self.stsize,self.stsize)
    return None

  def forward(self,iseq,xseq):
    """
    xseq [time,1,edim]: seq of embedded stimuli
    iseq [time,1]: seq indicating trial type (e.g. encode vs respond)
    """
    # reset memory
    self.EM_key,self.EM_value = [],[]
    # instruction path
    inst_seq = self.embed_instruct(iseq)
    inst_seq = self.i2inst(inst_seq).relu()
    ## initial states
    self.h_stim,self.c_stim = self.initst_stim
    self.h_main,self.c_main = self.initst_main 
    lstm_outputs = -tr.ones(len(xseq),1,self.stsize)
    ## trial loop 
    for tstep in range(len(xseq)):
      # print('\n-',iseq[tstep],xseq[tstep][0,0]) 
      ## sensory layer
      self.h_stim,self.c_stim = self.lstm_stim(xseq[tstep],(self.h_stim,self.c_stim))
      # self.h_stim = self.ff_stim(xseq[tstep])
      ## EM retrieval 
      if self.EMbool and (iseq[tstep] == self.resp_trial_flag):
        q = self.h_stim.detach().numpy()
        K = np.concatenate(self.EM_key)
        qksim = -pairwise_distances(q,K,metric='cosine').round(2).squeeze()
        retrieve_index = qksim.argmax()
        # print('EM:',qksim,retrieve_index)
        self.r_state = tr.Tensor(self.EM_value[retrieve_index])
        # transform and incorporate memory
        # self.r_state = self.ff_em2cell(self.r_state)
        self.c_main += self.r_state
      ## main layer
      lstm_main_in = tr.cat([inst_seq[tstep],self.h_stim],-1)
      self.h_main,self.c_main = self.lstm_main(lstm_main_in,(self.h_main,self.c_main))
      lstm_outputs[tstep] = self.h_main
      ## EM encoding
      if self.EMbool and (iseq[tstep] != self.resp_trial_flag):
        # em_key = concat ([h_stim,c_stim])
        self.EM_key.append(self.h_stim.detach().numpy())
        # try encoding h_main as value: 
        # this should produce same output as duing encoding phase
        self.EM_value.append(self.c_main.detach().numpy())
    ## output layer
    lstm_outputs = self.cell2outhid(lstm_outputs).relu()
    yhat_ulog = self.ff_hid2ulog(lstm_outputs)
    return yhat_ulog


class WMEM_PM(tr.nn.Module):

  def __init__(self,indim,pdim,stsize,outdim,EM=True,seed=132):
    super().__init__()
    # seed
    tr.manual_seed(seed)
    # params
    self.indim = indim
    self.stsize = stsize
    self.outdim = outdim
    self.pdim = stsize
    # layers
    self.stim2percept = tr.nn.Linear(indim,pdim) # stim2percept
    self.stim2percept_relu = tr.nn.ReLU()
    # Main LSTM CELL
    self.initial_state = tr.rand(2,1,self.stsize,requires_grad=True)
    self.lstm_main = tr.nn.LSTMCell(pdim,stsize)
    self.cell2outhid = tr.nn.Linear(stsize,stsize)
    self.cell2outhid_relu = tr.nn.ReLU()
    self.ff_hid2ulog = tr.nn.Linear(stsize,outdim)
    # Memory LSTM CELL
    self.rgate = tr.nn.Linear(pdim,stsize)
    self.sigm = tr.nn.Tanh()
    self.EM = EM
    return None

  def forward(self,xdata,EM=None):
    """ 
    input: xdata `(time,batch,embedding)`
    output: yhat `(time,batch,outdim)`
    """
    seqlen = xdata.shape[0]
    # inproj
    percept = self.stim2percept(xdata)
    percept = self.stim2percept_relu(percept)
    # compute retrieval similarities
    percept_pm_cue = percept[0]
    sim = (tr.cosine_similarity(percept_pm_cue,percept,dim=-1) + 1).detach()/2
    # sim = (tr.cosine_similarity(xdata[0],xdata,dim=-1) + 1).detach()/2
    self.sim = sim
    # unroll
    lstm_outs = -tr.ones(seqlen,1,self.stsize)
    ## PM cue encoding trial 
    # compute internal state for pm_cue 
    lstm_output,lstm_state = self.initial_state
    # lstm_output,lstm_state = tr.rand(2,1,self.stsize,requires_grad=False)
    lstm_output_pm,lstm_state_pm = self.lstm_main(percept_pm_cue,(lstm_output,lstm_state))
    lstm_outs[0] = lstm_output_pm
    ## task 
    lstm_output,lstm_state = lstm_output_pm,lstm_state_pm
    self.rgate_act = -tr.ones(seqlen,self.stsize)
    for t in range(1,seqlen):
      if EM == 'sim':
        # update state based on similarity to pm_state memory
        # mem_in = lstm_state_pm
        mem_h,mem_c = self.initial_state
        lstm_state = sim[t,0]*mem_h + (1-sim[t,0])*mem_h
        lstm_output = sim[t,0]*mem_c + (1-sim[t,0])*mem_c
      elif EM == 'gate':
        # rgate based retrieval
        rgate_act = self.rgate(percept[t,:,:])
        rgate_act = self.sigm(rgate_act)
        self.rgate_act[t] = rgate_act
        lstm_state = rgate_act*lstm_state_pm + lstm_state
      # compute cell prediction
      lstm_output,lstm_state = self.lstm_main(percept[t,:,:],(lstm_output,lstm_state))
      lstm_outs[t] = lstm_output
    # outporj
    yhat = self.cell2outhid(lstm_outs)
    yhat = self.cell2outhid_relu(yhat)
    yhat = self.ff_hid2ulog(yhat)
    return yhat

