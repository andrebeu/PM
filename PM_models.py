import torch as tr
import numpy as np

from sklearn.metrics import pairwise_distances


"""
assumes first input is the PM cue
"""

tr_uniform = lambda a,b,shape: tr.FloatTensor(*shape).uniform_(a,b)

class NetPureEM(tr.nn.Module):
  """ 
  arbitrary maps pureEM net, no LSTM
  first hid layers takes instruction and trial flag (no stim)
  stim useq to query EM
  """
  def __init__(self,wmsize=5,emsetting=1,seed=0,instdim=10,stimdim=10,debug=False):
    super().__init__()
    tr.manual_seed(seed)
    # params
    self.nmaps_max = 10
    self.instdim = instdim
    self.trdim = instdim
    self.wmdim = wmsize
    self.emdim = wmsize
    # outdims
    self.outdim = self.emdim+self.wmdim
    self.smdim = self.nmaps_max+1 # unit 0 not used
    # FF 
    self.embed_instruct = tr.nn.Embedding(self.nmaps_max+1,self.instdim)
    self.embed_trialflag = tr.nn.Embedding(2,self.trdim)
    self.ff_hid1 = tr.nn.Linear(self.instdim+self.trdim,self.wmdim,bias=False) 
    self.ff_hid2 = tr.nn.Linear(self.emdim+self.wmdim,self.outdim,bias=False) 
    # ouput
    self.ff_hid2ulog = tr.nn.Linear(self.outdim,self.smdim,bias=False)
    # EM setting 
    self.EMsetting = emsetting
    self.debug = debug
    self.emk = 'conj'
    self.store_states = False
    return None

  def forward(self,tseq,iseq,sseq):
    """ """
    ep_len = len(iseq)
    ntrials = 2
    trlen = int(ep_len/ntrials)
    # input pathways
    inst = self.embed_instruct(iseq)
    tflag = self.embed_trialflag(tseq)
    hid1 = self.ff_hid1(tr.cat([tflag,inst],-1))
    ## encode [hid1,x : hid1]
    emK_full = tr.cat([hid1,sseq],-1)
    for tr in range(ntrials):
      for tstep in range(tr*trlen,(tr+1)*trlen):
        emK_full


    print(hid1.shape) # [len(tseq),1,wimdim]
    # saving states
    statesL = []
    # initialize EM, LSTM, and outputs
    self.EM_key,self.EM_value = [],[]
    emoutputs = -tr.ones(ep_len,1,self.emdim)
    # loop over time
    for tstep in range(ep_len):
      if self.debug: 
        print()
      
      if self.store_states:
        states_t = np.stack([h_lstm.detach().numpy(),c_lstm.detach().numpy()],)
        statesL.append(states_t)
      # EM retrieval and encoding
      # em key
      emk = tr.cat([stim[tstep],h_lstm],-1)
      emv_encode = h_lstm
      if (iseq[tstep]==0):
        em_output_t = self.retrieve(emk)
        wm_output_t = h_lstm
        # wm_output_t = tr.zeros_like(h_lstm)
      else:
        self.encode(emk,emv_encode)
        em_output_t = tr.zeros_like(emv_encode)
        wm_output_t = h_lstm
      # output layer
      if self.debug:
        print('st',sseq[tstep,0,0].data)
        print('wm',h_lstm[0,0].data)
        print('em',em_output_t[0,0].data)
      outputs[tstep] = tr.cat([stim[tstep],wm_output_t,em_output_t],-1)
    ## output path
    if self.store_states:
      self.states = np.stack(statesL).squeeze()
    if tr.cuda.is_available():
      outputs = outputs.cuda()
    if self.out_hiddim:
      outputs = self.out_hid(outputs).relu()
    yhat_ulog = self.ff_hid2ulog(outputs)
    return yhat_ulog

  def retrieve(self,emquery):
    emquery = emquery.detach().cpu().numpy()
    EM_K = np.concatenate(self.EM_key)
    qkdist = pairwise_distances(emquery,EM_K,metric='cosine').squeeze().round(3)
    if self.debug:
      print('qkdist2',qkdist)
      print()
    retrieve_index = qkdist.argmin()
    em_output = tr.Tensor(self.EM_value[retrieve_index])
    if tr.cuda.is_available():
      em_output = em_output.cuda()
    return em_output

  def encode(self,emk,emv):
    emk = emk.detach().cpu().numpy()
    emv = emv.detach().cpu().numpy()
    self.EM_key.append(emk)
    self.EM_value.append(emv)
    return None


class NetBarCode(tr.nn.Module):
  """ 
  arbitrary maps EM net 
  """
  def __init__(self,wmsize=5,emsetting=1,seed=0,instdim=10,stimdim=10,debug=False):
    super().__init__()
    tr.manual_seed(seed)
    # params
    self.nmaps_max = 20
    self.instdim = instdim
    self.stimdim = stimdim
    self.wmdim = wmsize
    self.hiddim = wmsize
    self.emdim = self.hiddim
    ##
    self.outdim = self.hiddim+self.wmdim+self.emdim
    self.smdim = self.nmaps_max+1 # unit 0 not used
    # input 
    self.embed_instruct = tr.nn.Embedding(self.nmaps_max+1,self.instdim)
    self.ff_stim = tr.nn.Linear(self.stimdim,self.stimdim,bias=False)
    # hid ff and lstm
    self.ff_hid = tr.nn.Linear(self.stimdim+self.instdim,self.hiddim,bias=False) 
    self.lstm = tr.nn.LSTMCell(self.instdim+self.stimdim,self.wmdim)
    self.init_lstm = tr.nn.Parameter(tr.rand(2,1,self.wmdim),requires_grad=True)
    # ouput
    self.out_hid = tr.nn.Linear(self.outdim,self.outdim,bias=False)
    self.ff_hid2ulog = tr.nn.Linear(self.outdim,self.smdim,bias=False)
    # EM setting 
    self.emk_weights = [2,0]
    self.EMsetting = emsetting
    self.debug = debug
    self.store_states = False
    return None

  def forward(self,iseq,sseq):
    """ """
    ep_len = len(iseq)
    # input pathways
    inst = self.embed_instruct(iseq)
    stim = self.ff_stim(sseq).relu()
    # cat and feed forward percept
    percept = tr.cat([inst,stim],-1)
    hid_ff = self.ff_hid(percept).relu()
    # saving states
    statesL = []
    # initialize EM, LSTM, and outputs
    self.EM_key,self.EM_value = [],[]
    outputs = -tr.ones(ep_len,1,self.outdim)
    h_lstm,c_lstm = self.init_lstm
    # loop over time
    for tstep in range(ep_len):
      if self.debug: 
        print()
      h_lstm,c_lstm = self.lstm(percept[tstep],(h_lstm,c_lstm))
      if self.store_states:
        states_t = np.stack([h_lstm.detach().cpu().numpy(),c_lstm.detach().cpu().numpy()])
        statesL.append(states_t)
      ## EM retrieval and encoding
      if (self.EMsetting>0): 
        # em key (encode and query)
        emkL = [stim[tstep],h_lstm]
        # test phase
        if (iseq[tstep]==0):
          em_output_t = self.retrieve_conj(emkL)
          wm_output_t = h_lstm
        # instruction phase
        else:
          # em value
          emv = hid_ff[tstep]
          self.encode_conj(emkL,emv)
          em_output_t = emv
          wm_output_t = h_lstm
      else: 
        em_output_t = tr.zeros_like(h_lstm)
        wm_output_t = h_lstm
      # output layer
      outputs[tstep] = tr.cat([hid_ff[tstep],wm_output_t,em_output_t],-1)
    ## output path
    if self.store_states:
      self.states = np.stack(statesL).squeeze()
    if tr.cuda.is_available():
      outputs = outputs.cuda()
    outputs = self.out_hid(outputs).relu()
    yhat_ulog = self.ff_hid2ulog(outputs)
    return yhat_ulog

  def encode_conj(self,emkL,emv):
    ''' conjunctive keys '''
    emkL = [emk.detach().cpu().numpy() for emk in emkL]
    emv = emv.detach().cpu().numpy()
    self.EM_key.append(emkL)
    self.EM_value.append(emv)
    return None

  def retrieve_conj(self,emqueryL):
    ''' conjunctive keys '''
    ## loop over elements of conjunction 
    # (dimensions of memory - each dimension is a vector)
    # calculate qk_distances on each dimension 
    emk_weights = self.emk_weights
    qkdist = -np.ones([len(emqueryL),len(self.EM_key)])
    for emk_dim in range(len(emqueryL)):
      emquery = emqueryL[emk_dim].detach().cpu().numpy()
      emK = np.concatenate([emk[emk_dim] for emk in self.EM_key],0)
      qkdist[emk_dim] = pairwise_distances(emquery,emK,metric='cosine').squeeze().round(3)
    # combine qkdist of different dimensions with weights
    qkdist = np.dot(emk_weights,qkdist)
    # retrieve
    retrieve_index = qkdist.argmin()
    memory = tr.Tensor(self.EM_value[retrieve_index])
    if tr.cuda.is_available(): memory = memory.cuda()
    return memory

  def retrieve(self,emquery):
    emquery = emquery.detach().cpu().numpy()
    EM_K = np.concatenate(self.EM_key)
    qkdist = pairwise_distances(emquery,EM_K,metric='cosine').squeeze().round(3)
    retrieve_index = qkdist.argmin()
    em_output = tr.Tensor(self.EM_value[retrieve_index])
    if tr.cuda.is_available():
      em_output = em_output.cuda()
    return em_output

  def encode(self,emk,emv):
    emk = emk.detach().cpu().numpy()
    emv = emv.detach().cpu().numpy()
    self.EM_key.append(emk)
    self.EM_value.append(emv)
    return None


class NetAMEM(tr.nn.Module):
  """ 
  arbitrary maps EM net 
  """
  def __init__(self,stsize=25,emsetting=1,wmsetting=1,seed=0,instdim=10,stimdim=10):
    super().__init__()
    tr.manual_seed(seed)
    # params
    self.nmaps_max = 10
    # net dims
    self.instdim = instdim
    self.stimdim = stimdim
    self.wmdim = stsize
    self.emdim = stsize
    self.outdim = self.wmdim+self.emdim
    self.smdim = self.nmaps_max+1 # unit 0 not used
    # instruction in pathway
    self.embed_instruct = tr.nn.Embedding(self.nmaps_max+1,self.instdim)
    self.i2inst = tr.nn.Linear(self.instdim,self.instdim,bias=False) 
    # stimulus in pathway
    self.ff_stim = tr.nn.Linear(self.stimdim,self.stimdim,bias=False)
    # main LSTM
    self.lstm1 = tr.nn.LSTMCell(self.instdim+self.stimdim,self.wmdim)
    self.lstm2 = tr.nn.LSTMCell(self.wmdim,self.wmdim)
    self.init_lstm1 = tr.nn.Parameter(tr.rand(2,1,self.wmdim),requires_grad=True)
    self.init_lstm2 = tr.nn.Parameter(tr.rand(2,1,self.wmdim),requires_grad=True)
    # output layers
    self.cell2outhid = tr.nn.Linear(self.outdim,self.outdim,bias=False)
    self.ff_hid2ulog = tr.nn.Linear(self.outdim,self.smdim,bias=False)
    # EM setting 
    self.EMsetting = emsetting
    self.WMsetting = wmsetting
    self.deep = True
    return None

  def forward(self,iseq,sseq,store_states=False):
    """ """
    ep_len = len(iseq)
    # instruction pathway
    inst_seq = self.embed_instruct(iseq)
    stim_seq = sseq
    if self.deep:
      inst_seq = self.i2inst(inst_seq).relu()
      # stim in path 
      stim_seq = self.ff_stim(stim_seq).relu()
    # init EM
    self.EM_key,self.EM_value = [],[]
    # save lstm states
    cstateL,hstateL = [],[]
    # collect outputs
    outputs = -tr.ones(ep_len,1,self.outdim)
    # LSTM unroll
    h_lstm1,c_lstm1 = self.init_lstm1
    h_lstm2,c_lstm2 = self.init_lstm2 
    for tstep in range(ep_len):
      # print(tstep,iseq[tstep],sseq[tstep,0,0])
      ## LSTM 1
      lstm1_in = tr.cat([inst_seq[tstep],stim_seq[tstep]],-1)
      h_lstm1,c_lstm1 = self.lstm1(lstm1_in,(h_lstm1,c_lstm1))
      wm_output_t = h_lstm1
      ## EM retrieval and encoding
      if (self.EMsetting>0): 
        # em key
        emk = emv = h_lstm1
        if (iseq[tstep]==0): 
          em_output_t = self.retrieve(emk)
        else:
          self.encode(emk,emv)
          em_output_t = tr.zeros_like(h_lstm1)
      else: 
        em_output_t = tr.zeros_like(h_lstm1)
      ## deep LSTM 
      if self.WMsetting==2:
        lstm2_in = h_lstm1 
        wm_output_t,c_lstm2 = self.lstm2(lstm2_in,(wm_output_t,c_lstm2))
      ## output layer
      outputs[tstep] = tr.cat([wm_output_t,em_output_t],-1)
    ## output path
    if tr.cuda.is_available():
      outputs = outputs.cuda()
    if self.deep:
      outputs = self.cell2outhid(outputs).relu()
    yhat_ulog = self.ff_hid2ulog(outputs)
    # save lstm states
    self.cstates,self.hstates = np.array(cstateL),np.array(hstateL)
    return yhat_ulog

  def retrieve(self,emquery):
    emquery = emquery.detach().cpu().numpy()
    EM_K = np.concatenate(self.EM_key)
    qksim = -pairwise_distances(emquery,EM_K,metric='cosine').round(2).squeeze()
    retrieve_index = qksim.argmax()
    em_output = tr.Tensor(self.EM_value[retrieve_index])
    if tr.cuda.is_available():
      em_output = em_output.cuda()
    return em_output

  def encode(self,emk,emv):
    emk = emk.detach().cpu().numpy()
    emv = emv.detach().cpu().numpy()
    self.EM_key.append(emk)
    self.EM_value.append(emv)
    return None


class NetDualLSTMToy(tr.nn.Module):
  """ 
  two LSTMS 
    one processes instruction phase to compute an EM embedding
    another takes EM embedding and test phase stim to respond
  """

  def __init__(self,stsize1=20,stsize2=35,seed=0,deep_lstm2=True):
    
    super().__init__()
    tr.manual_seed(seed)
    # params
    self.nmaps = 10
    # net dims
    self.instdim = 10
    self.stimdim = 10
    self.stsize_lstm1 = stsize1
    self.stsize_lstm2 = stsize2
    self.outdim = 10 # 2 OG units, 8 PM units
    # instruction in pathway
    self.embed_instruct = tr.nn.Embedding(self.nmaps+2,self.instdim)
    self.i2inst = tr.nn.Linear(self.instdim,self.instdim,bias=True) 
    # stimulus in pathway
    self.ff_stim = tr.nn.Linear(self.stimdim,self.stimdim,bias=True)
    # main LSTM
    self.lstm1 = tr.nn.LSTMCell(self.instdim+self.stimdim,self.stsize_lstm1)
    self.lstm2 = tr.nn.LSTMCell(self.stimdim+self.stsize_lstm1,self.stsize_lstm2)
    self.lstm22 = tr.nn.LSTMCell(self.stsize_lstm2,self.stsize_lstm2)
    self.init_lstm1 = tr.nn.Parameter(tr.rand(2,1,self.stsize_lstm1),requires_grad=True)
    self.init_lstm2 = tr.nn.Parameter(tr.rand(2,1,self.stsize_lstm2),requires_grad=True)
    self.init_lstm22 = tr.nn.Parameter(tr.rand(2,1,self.stsize_lstm2),requires_grad=True)
    if deep_lstm2: print('two LSTM layers')
    self.deep_lstm2 = deep_lstm2
    # output layers
    self.cell2outhid = tr.nn.Linear(self.stsize_lstm2,self.stsize_lstm2,bias=True)
    self.ff_hid2ulog = tr.nn.Linear(self.stsize_lstm2,self.outdim,bias=True)
    return None

  def forward(self,iseq,sseq,store_states=False):
    """ """
    ep_len = len(iseq)
    # input layer inst in path
    inst_seq = self.embed_instruct(iseq)
    inst_seq = self.i2inst(inst_seq).relu() 
    stim_seq = self.ff_stim(sseq).relu()
    # init EM
    self.EM_key,self.EM_value = [],[]
    # save lstm states
    cstateL,hstateL = [],[]
    # collect outputs
    WM_outputs = []
    # LSTM unroll
    h_lstm1,c_lstm1 = self.init_lstm1
    h_lstm2,c_lstm2 = self.init_lstm2 
    h_lstm22,c_lstm22 = self.init_lstm22 
    for tstep in range(ep_len):
      lstm1_in = tr.cat([inst_seq[tstep],stim_seq[tstep]],-1)
      h_lstm1,c_lstm1 = self.lstm1(lstm1_in,(h_lstm1,c_lstm1))
      ## learn phase
      if (iseq[tstep]!=0):
        context_em = h_lstm1
      ## test phase
      else:
        em_embed = context_em
        lstm2_in = tr.cat([em_embed,stim_seq[tstep]],-1)
        h_lstm2,c_lstm2 = self.lstm2(lstm2_in,(h_lstm2,c_lstm2))
        h_lstm22,c_lstm22 = self.lstm22(h_lstm2,(h_lstm22,c_lstm22))
        if self.deep_lstm2:
          WM_outputs.append(h_lstm22)
        else:
          WM_outputs.append(h_lstm2)
    ## output path
    WM_outputs = tr.stack(WM_outputs)
    WM_outputs = self.cell2outhid(WM_outputs).relu()
    yhat_ulog = self.ff_hid2ulog(WM_outputs)
    # save lstm states
    # self.cstates,self.hstates = np.array(cstateL),np.array(hstateL)
    return yhat_ulog


class NetHlstmEM(tr.nn.Module):
  """ 
  hierarchical lstm
  EM injected into LSTM cell state
  """

  def __init__(self,stsize=25,emsetting=1,seed=0):
    super().__init__()
    tr.manual_seed(seed)
    # params
    self.nmaps = 10
    # net dims
    self.instdim = 10
    self.stimdim = 10
    self.stsize = stsize
    self.stsize_em = 10
    self.outdim = 10 # 2 OG units, 8 PM units
    # instruction in pathway
    self.embed_instruct = tr.nn.Embedding(self.nmaps+1,self.instdim)
    self.i2inst = tr.nn.Linear(self.instdim,self.instdim,bias=True) 
    # stimulus in pathway
    self.ff_stim = tr.nn.Linear(self.stimdim,self.stimdim,bias=True)
    # main LSTM
    self.lstm1 = tr.nn.LSTMCell(self.stimdim+self.instdim,self.stsize)
    self.lstm2 = tr.nn.LSTMCell(self.stsize,self.stsize)
    self.init_lstm1 = tr.nn.Parameter(tr.rand(2,1,self.stsize),requires_grad=True)
    self.init_lstm2 = tr.nn.Parameter(tr.rand(2,1,self.stsize),requires_grad=True)
    # EM LSTM
    self.lstm_em = tr.nn.LSTMCell(self.stimdim+self.instdim,self.stsize)
    self.init_lstm_em = tr.nn.Parameter(tr.rand(2,1,self.stsize),requires_grad=True)
    # output layers
    self.cell2outhid = tr.nn.Linear(self.stsize,self.stsize,bias=True)
    self.ff_hid2ulog = tr.nn.Linear(self.stsize,self.outdim,bias=True)
    # EM setting 
    self.EMsetting = emsetting
    self.WMsetting = 1
    return None

  def forward(self,iseq,sseq,store_states=False):
    """ """
    ep_len = len(iseq)
    # inst in path
    inst_seq = self.embed_instruct(iseq)
    inst_seq = self.i2inst(inst_seq).relu()
    # stim in path 
    stim_seq = self.ff_stim(sseq).relu()
    # init EM
    self.EM_key,self.EM_value = [],[]
    # save lstm states
    cstateL,hstateL = [],[]
    # collect outputs
    WM_outputs = -tr.ones(ep_len,1,self.stsize+self.emdim)
    # LSTM unroll
    h_lstm1,c_lstm1 = self.init_lstm1
    h_lstm2,c_lstm2 = self.init_lstm2 
    h_em,c_em = self.init_lstm_em
    for tstep in range(ep_len):
      ## LSTM 1
      lstm1_in = tr.cat([inst_seq[tstep],stim_seq[tstep]],-1)
      h_lstm1,c_lstm1 = self.lstm1(lstm1_in,(h_lstm1,c_lstm1))
      ## LSTM EM
      lstm_em_in = tr.cat([inst_seq[tstep],stim_seq[tstep]],-1)
      h_em,c_em = self.lstm_em(lstm_em_in,(h_em,c_em))
      ## EM retrieval and encoding
      if (self.EMsetting>0): 
        # em key
        emk = emv = h_lstm1
        if (iseq[tstep]==0): 
          # print('retrieve')
          em_output = self.retrieve(emk)
        else:
          # print('encode')
          self.encode(emk,emv)
          em_output = tr.zeros_like(h_lstm1)
      else: 
        em_output = tr.zeros_like(h_lstm1)
      ## LSTM 2
      lstm2_in = h_lstm1 + em_output
      h_lstm2,c_lstm2 = self.lstm2(lstm2_in,(h_lstm2,c_lstm2))
      ## output layer
      WM_outputs[tstep] = h_lstm2
    ## output path
    WM_outputs = self.cell2outhid(WM_outputs).relu()
    yhat_ulog = self.ff_hid2ulog(WM_outputs)
    # save lstm states
    self.cstates,self.hstates = np.array(cstateL),np.array(hstateL)
    return yhat_ulog

  def retrieve(self,emquery):
    emquery = emquery.detach().numpy()
    EM_K = np.concatenate(self.EM_key)
    qksim = -pairwise_distances(emquery,EM_K,metric='cosine').round(2).squeeze()
    retrieve_index = qksim.argmax()
    h_memory = tr.Tensor(self.EM_value[retrieve_index])
    return em_output

  def encode(self,emk,emv):
    emk = emk.detach().numpy()
    emv = emv.detach().numpy()
    self.EM_key.append(emk)
    self.EM_value.append(emv)
    return None


class NetDualPM(tr.nn.Module):
  """ EM and WM have separate pathways
  """

  def __init__(self,stsize=25,emsetting=1,seed=0):
    super().__init__()
    tr.manual_seed(seed)
    # params
    self.nmaps = 10
    # net dims
    self.instdim = 10
    self.stimdim = 10
    self.stsize = stsize
    self.emdim = stsize
    self.outdim = 10 # 2 OG units, 8 PM units
    # instruction in pathway
    self.embed_instruct = tr.nn.Embedding(self.nmaps+1,self.instdim)
    self.i2inst = tr.nn.Linear(self.instdim,self.instdim,bias=True) 
    # stimulus in pathway
    self.ff_stim = tr.nn.Linear(self.stimdim,self.stimdim,bias=True)
    # main LSTM
    self.lstm_main = tr.nn.LSTMCell(self.stimdim+self.instdim,self.stsize)
    # self.init_st_main = tr.rand(2,1,self.stsize,requires_grad=True)
    self.init_st_main = tr.nn.Parameter(tr.rand(2,1,self.stsize),requires_grad=True)
    # output layers
    self.cell2outhid = tr.nn.Linear(self.stsize+self.emdim,self.stsize,bias=True)
    self.ff_hid2ulog = tr.nn.Linear(self.stsize,self.outdim,bias=True)
    # EM setting {0:no_em,1:em_nogate,2:em_gate}
    self.EMsetting = emsetting
    self.WMsetting = 1
    return None

  def forward(self,iseq,sseq,store_states=False):
    """ """
    ep_len = len(iseq)
    self.rgate_actL = []
    # inst in path
    inst_seq = self.embed_instruct(iseq)
    inst_seq = self.i2inst(inst_seq).relu()
    # stim in path 
    stim_seq = self.ff_stim(sseq).relu()
    # init EM
    self.EM_key,self.EM_value = [],[]
    # save lstm states
    cstateL,hstateL = [],[]
    # LSTM unroll
    cell_outputs = -tr.ones(ep_len,1,self.stsize+self.emdim)
    h_main,c_main = self.init_st_main 
    for tstep in range(ep_len):
      # forward prop LSTM
      lstm_main_in = tr.cat([inst_seq[tstep],stim_seq[tstep]],-1)
      h_main,c_main = self.lstm_main(lstm_main_in,(h_main,c_main))
      # save lstm states
      if store_states:
        cstateL.append(c_main.detach().numpy().squeeze())
        hstateL.append(h_main.detach().numpy().squeeze())
      if (self.EMsetting==1): 
        # form memory key
        emk = np.concatenate([
                # stim_seq[tstep].detach().numpy(),
                h_main.detach().numpy()
                ],-1)
        if (iseq[tstep]==0): 
          # print('retrieve')
          emquery = emk
          EM_K = np.concatenate(self.EM_key)
          qksim = -pairwise_distances(emquery,EM_K,metric='cosine').round(2).squeeze()
          retrieve_index = qksim.argmax()
          h_memory = tr.Tensor(self.EM_value[retrieve_index])
        else:
          # print('encode')
          self.EM_value.append(h_main.detach().numpy())
          self.EM_key.append(emk)
          h_memory = tr.zeros_like(h_main)
      else: 
        h_memory = tr.zeros_like(h_main)
      if self.WMsetting == False:
        h_main = tr.zeros_like(h_main)
      cell_outputs[tstep] = tr.cat([h_main,h_memory],-1)
    ## output path
    cell_outputs = self.cell2outhid(cell_outputs).relu()
    yhat_ulog = self.ff_hid2ulog(cell_outputs)
    # save lstm states
    self.cstates,self.hstates = np.array(cstateL),np.array(hstateL)
    return yhat_ulog


class PINet(tr.nn.Module):

  def __init__(self,stimdim,stsize,outdim,nmaps,EMbool=True,seed=132):
    super().__init__()
    # seed
    tr.manual_seed(seed)
    ## layer sizes 
    self.stimdim = stimdim
    self.instdim = stimdim
    self.nmaps = nmaps
    self.resp_trial_flag = nmaps-1
    self.stsize = stsize
    self.outdim = outdim
    self.emdim = stsize
    ## instruction layer
    self.embed_instruct = tr.nn.Embedding(self.nmaps,self.instdim)
    self.i2inst = tr.nn.Linear(self.instdim,self.instdim,bias=False) 
    ## sensory layer
    # self.stim_emat = tr_uniform(0,1,[self.nmaps,self.stimdim])
    self.ff_stim = tr.nn.Linear(self.stimdim,self.stimdim,bias=False)
    ## Main LSTM CELL
    self.lstm_main = tr.nn.LSTMCell(self.stimdim+self.instdim,self.stsize)
    self.initst_main = tr.rand(2,1,self.stsize,requires_grad=True)
    ## out proj
    self.cell2outhid = tr.nn.Linear(self.stsize+self.emdim,self.stsize,bias=False)
    self.ff_hid2ulog = tr.nn.Linear(self.stsize,self.outdim,bias=False)
    ## Episodic memory
    self.EMbool = EMbool
    self.WMbool = True
    self.EM_key = []
    self.EM_value = []
    # self.ff_em2cell = tr.nn.Linear(self.stsize,self.stsize)
    return None


  def forward(self,iseq,xseq):
    """
    xseq [time,1,edim]: seq of int stimuli
    iseq [time,1]: seq indicating trial type (e.g. encode vs respond)
    """
    self.cstateL,self.hstateL = [],[]
    # reset memory
    self.EM_key,self.EM_value = [],[]
    # instruction path
    inst_seq = self.embed_instruct(iseq)
    inst_seq = self.i2inst(inst_seq).relu()
    ## stimulus embedding
    # xseq = self.stim_emat[xseq]
    ## initial states
    # self.h_stim,self.c_stim = self.initst_stim
    self.h_main,self.c_main = self.initst_main 
    cell_outputs = -tr.ones(len(xseq),1,self.stsize+self.emdim)
    ## trial loop 
    for tstep in range(len(xseq)):
      # print('\n-',iseq[tstep],xseq[tstep][0,0]) 
      ## sensory layer
      self.h_stim = self.ff_stim(xseq[tstep])
      ## main layer
      lstm_main_in = tr.cat([inst_seq[tstep],self.h_stim],-1)
      self.h_main,self.c_main = self.lstm_main(lstm_main_in,(self.h_main,self.c_main))
      self.hstateL.append(self.h_main.detach().numpy().squeeze())
      self.cstateL.append(self.c_main.detach().numpy().squeeze())
      ## EM retrieval 
      if self.EMbool and (iseq[tstep] == self.resp_trial_flag):
        query = np.concatenate([
                  self.h_stim.detach().numpy(),
                  self.h_main.detach().numpy()
                  ],-1)
        EM_K = np.concatenate(self.EM_key)
        qksim = -pairwise_distances(query,EM_K,metric='cosine').round(2).squeeze()
        retrieve_index = qksim.argmax()
        self.r_state = tr.Tensor(self.EM_value[retrieve_index])
      ## EM encoding
      elif self.EMbool and (iseq[tstep] != self.resp_trial_flag):
        self.r_state = tr.zeros_like(self.h_main)
        em_key = np.concatenate([
                  self.h_stim.detach().numpy(),
                  self.h_main.detach().numpy()
                  ],-1)
        self.EM_key.append(em_key)
        self.EM_value.append(self.h_main.detach().numpy())
      else:
        self.r_state = tr.zeros_like(self.h_main)
      ### timestep output
      if self.WMbool == False:
        self.h_main = tr.zeros_like(self.h_main)
      cell_outputs[tstep] = tr.cat([self.h_main,self.r_state],-1)
    ## output layer
    cell_outputs = self.cell2outhid(cell_outputs).relu()
    yhat_ulog = self.ff_hid2ulog(cell_outputs)
    ## storing states
    self.hstates = np.array(self.hstateL)
    self.cstates = np.array(self.cstateL)
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

