import torch as tr
import numpy as np


"""
assumes first input is the PM cue
"""

class PINet(tr.nn.Module):

  def __init__(self,stimdim,stsize,outdim,ninstructs,EM=False,seed=132):
    super().__init__()
    # seed
    tr.manual_seed(seed)
    # layer sizes
    self.stimdim = stimdim
    self.instdim = stimdim
    self.ninstructs = ninstructs
    self.stsize = stsize
    self.outdim = outdim
    # embed instructions
    self.embed_instruct = tr.nn.Embedding(self.ninstructs,self.instdim)
    self.i2inst = tr.nn.Linear(self.instdim,self.instdim) 
    self.i2inst_relu = tr.nn.ReLU()
    # project stim2stim
    self.x2stim = tr.nn.Linear(self.stimdim,self.stimdim) # x2stim
    self.x2stim_relu = tr.nn.ReLU()
    # Main LSTM CELL
    self.initial_state = tr.rand(2,1,self.stsize,requires_grad=True) ## previously reqgrad=False
    self.cell_main = tr.nn.LSTMCell(self.stimdim+self.instdim,self.stsize)
    # out proj
    self.cell2outhid = tr.nn.Linear(self.stsize,self.stsize)
    self.cell2outhid_relu = tr.nn.ReLU()
    self.ff_out2 = tr.nn.Linear(self.stsize,self.outdim)
    # Memory LSTM CELL
    self.EM = EM
    return None

  def forward(self,iseq,xseq):
    """
    xseq [time,1,edim]: seq of embedded stimuli
    iseq [time,1]: seq indicating trial type (e.g. encode vs respond)
    """
    # instruction path
    inst_seq = self.embed_instruct(iseq) 
    inst_seq = self.i2inst(inst_seq)
    inst_seq = self.i2inst_relu(inst_seq)
    # sensory path
    stim_seq = self.x2stim(xseq)
    stim_seq = self.x2stim_relu(stim_seq)
    # percept
    inseq = tr.cat([inst_seq,stim_seq],-1)
    # cell
    self.h_state,self.c_state = self.initial_state ## previous h,c = h.data,c.data
    lstm_outputs = -tr.ones(len(inseq),1,self.stsize)
    for tstep in range(len(inseq)):
      self.h_state,self.c_state = self.cell_main(inseq[tstep],(self.h_state,self.c_state))
      lstm_outputs[tstep] = self.h_state
    # output
    lstm_outputs = self.cell2outhid(lstm_outputs)
    lstm_outputs = self.cell2outhid_relu(lstm_outputs)
    yhat_ulog = self.ff_out2(lstm_outputs)
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
    self.cell_main = tr.nn.LSTMCell(pdim,stsize)
    self.cell2outhid = tr.nn.Linear(stsize,stsize)
    self.cell2outhid_relu = tr.nn.ReLU()
    self.ff_out2 = tr.nn.Linear(stsize,outdim)
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
    lstm_output_pm,lstm_state_pm = self.cell_main(percept_pm_cue,(lstm_output,lstm_state))
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
      lstm_output,lstm_state = self.cell_main(percept[t,:,:],(lstm_output,lstm_state))
      lstm_outs[t] = lstm_output
    # outporj
    yhat = self.cell2outhid(lstm_outs)
    yhat = self.cell2outhid_relu(yhat)
    yhat = self.ff_out2(yhat)
    return yhat

