#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 19:01:44 2020

@author: Buddhi Wickramasinghe

# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by - Srikanth Madikeri (2017)
# Based on the code from Ivan Himawan (QUT, Brisbane)
"""



from __future__ import print_function, division
import os
import sys
import math
import torch
# import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle as pickle
# import h5py
import hdf5storage
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import os


###### Loading framed dataset
# frame and save signals as .npy (frame_signals) -> data class
class ASVSpoofTrainData(Dataset):

    def __init__(self, transform=None):
        labelsData = hdf5storage.loadmat('data/labels_trainSel.mat')
        self.classes = labelsData['labels_trainSel']
        self.num_obj = len(self.classes)
        self.classes = torch.from_numpy(np.array(self.classes)).long()

    def __len__(self):
        return self.num_obj

    def __getitem__(self, idx):
        if idx > self.num_obj:
            raise IndexError()

        # printing environment variables
        #  print(os.environ)
        #  home_dir = os.environ['CONDA_DEFAULT_ENV']
        #  print(home_dir)
        mat = np.load('data/TrainSel/tr' + str(idx) + '.npy')
        return (torch.from_numpy(mat).double(), self.classes[idx])


class ASVSpoofTrainDataG(Dataset):

    def __init__(self, transform=None):
        labelsData = hdf5storage.loadmat('labels_trainGSel.mat')
        self.classes = labelsData['labels_trainGSel']
        self.num_obj = len(self.classes)
        self.classes = torch.from_numpy(np.array(self.classes)).long()

    def __len__(self):
        return self.num_obj

    def __getitem__(self, idx):
        if idx > self.num_obj:
            raise IndexError()

        mat = np.load('data/TrainRev/trG' + str(idx) + '.npy')
        return (torch.from_numpy(mat).double(), self.classes[idx])


class ASVSpoofTrainDataS(Dataset):

    def __init__(self, transform=None):
        labelsData = hdf5storage.loadmat('labels_trainSSel.mat')
        self.classes = labelsData['labels_trainSSel']
        self.num_obj = len(self.classes)
        self.classes = torch.from_numpy(np.array(self.classes)).long()

    def __len__(self):
        return self.num_obj

    def __getitem__(self, idx):
        if idx > self.num_obj:
            raise IndexError()

        mat = np.load('data/TrainRev/trS' + str(idx) + '.npy')
        return (torch.from_numpy(mat).double(), self.classes[idx])


class ASVSpoofDevData(Dataset):

    def __init__(self, transform=None):
        labelsData = hdf5storage.loadmat('data/labels_devSel.mat')
        self.classes = labelsData['labels_devSel']
        self.num_obj = len(self.classes)
        self.classes = torch.from_numpy(np.array(self.classes)).long()

    def __len__(self):
        return self.num_obj

    def __getitem__(self, idx):
        if idx > self.num_obj:
            raise IndexError()

        mat = np.load('data/DevSel/eval' + str(idx) + '.npy')
        return (torch.from_numpy(mat).double(), self.classes[idx])


class ASVSpoofEvalData(Dataset):

    def __init__(self, transform=None):
        labelsData = hdf5storage.loadmat('labels_devSel.mat')
        self.classes = labelsData['labels_devSel']
        self.num_obj = len(self.classes)
        self.classes = torch.from_numpy(np.array(self.classes)).long()

    def __len__(self):
        return self.num_obj

    def __getitem__(self, idx):
        if idx > self.num_obj:
            raise IndexError()

        mat = np.load('data/EvalSel/eval' + str(idx) + '.npy')
        return (torch.from_numpy(mat).double(), self.classes[idx])


# if __name__ == '__main__':
# d = ASVSpoofEvalData()
# dl = DataLoader(d, batch_size=1)
# for (x, l) in dl:
#     print(x.shape)
