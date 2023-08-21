"""
Code from: https://github.com/mravanelli/SincNet
Modified by: Buddhi Wickramasinghe
"""
import configparser as ConfigParser
from optparse import OptionParser
import numpy as np

def ReadList(list_file):
 f=open(list_file,"r")
 lines=f.readlines()
 list_sig=[]
 for x in lines:
    list_sig.append(x.rstrip())
 f.close()
 return list_sig


def read_conf(cfg_file=None):
 parser=OptionParser()
 parser.add_option("--cfg") # Mandatory
 (options,args)=parser.parse_args()
 if cfg_file is None:
  cfg_file=options.cfg
 if options.cfg is None and cfg_file is None:
  cfg_file='cfg/config_file.cfg'
 Config = ConfigParser.ConfigParser()
 Config.read(cfg_file)


 #[data]
 options.tr_lst=Config.get('data', 'tr_lst')
 options.te_lst=Config.get('data', 'te_lst')
 options.labTr_dict=Config.get('data', 'labTr_dict')
 options.labTe_dict=Config.get('data', 'labTe_dict')
 options.data_folder=Config.get('data', 'data_folder')
 options.output_folder=Config.get('data', 'output_folder')
 options.pt_file=Config.get('data', 'pt_file')

 #[windowing]
 options.fs=Config.get('windowing', 'fs')
 options.cw_len=Config.get('windowing', 'cw_len')
 options.cw_shift=Config.get('windowing', 'cw_shift')
 options.num_frames = Config.get('windowing', 'num_frames')

 #[gabor]
 options.cnn_N_filt=Config.get('gabor', 'cnn_N_filt')
 options.cnn_len_filt=Config.get('gabor', 'cnn_len_filt')

 #[optimization]
 options.lr=Config.get('optimization', 'lr')
 options.batch_size=Config.get('optimization', 'batch_size')
 options.N_epochs=Config.get('optimization', 'N_epochs')
 options.N_batches=Config.get('optimization', 'N_batches')
 options.seed=Config.get('optimization', 'seed')
 options.device = Config.get('optimization', 'device')
 return options


def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError
