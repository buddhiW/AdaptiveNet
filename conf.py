"""
Code from: https://github.com/mravanelli/SincNet
Modified by: Buddhi Wickramasinghe
"""
from data_io import ReadList,read_conf
import numpy as np
# Reading cfg file
options=read_conf()

#[data]
tr_lst=options.tr_lst
te_lst=options.te_lst
pt_file=options.pt_file
class_dictTr_file=options.labTr_dict
class_dictTe_file=options.labTe_dict
data_folder=options.data_folder+'/'
output_folder=options.output_folder

#[windowing]
fs=int(options.fs)
cw_len=int(options.cw_len)
cw_shift=int(options.cw_shift)
num_frames=int(options.num_frames)

#[gabor]
cnn_N_filt=list(map(int, options.cnn_N_filt.split(',')))
cnn_len_filt=list(map(int, options.cnn_len_filt.split(',')))

#[optimization]
lr=float(options.lr)
batch_size=int(options.batch_size)
N_epochs=int(options.N_epochs)
N_batches=int(options.N_batches)
seed=int(options.seed)
device=str(options.device)

# training list
wav_lst_tr=ReadList(tr_lst)
snt_tr=len(wav_lst_tr)

# test list
wav_lst_te=ReadList(te_lst)
snt_te=len(wav_lst_te)

# Converting context and shift in samples
wlen=int(fs*cw_len/1000.00)
wshift=int(fs*cw_shift/1000.00)

# Batch_dev
Batch_dev=128

# Loading label dictionary
lab_dictTr=np.load(class_dictTr_file, allow_pickle=True).item()
lab_dictTe=np.load(class_dictTe_file, allow_pickle=True).item()

# Initialization of the minibatch (batch_size,[0=>x_t,1=>x_t+N,1=>random_samp])
sig_batch=np.zeros([batch_size,wlen])
lab_batch=np.zeros(batch_size)
#out_dim = class_lay[0]