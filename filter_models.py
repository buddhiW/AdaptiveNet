"""
Created on Nov 15 2019

Author: Buddhi Wickramasinghe

"""
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
from conf import *


class gabor_fixedResponse(nn.Module):

    def __init__(self, N_filt,Filt_dim,Batch_Size,fs,fc):
        super(gabor_fixedResponse,self).__init__()

        # Mel Initialization of the filterbanks
        low_freq_mel = 80
        high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, N_filt)  # Equally spaced in Mel scale
        f_cos = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz

        fdelta = (fs/2)/(N_filt)
        f_cos = np.arange(fdelta/2, fs/2, fdelta)

        b1 = np.roll(f_cos, 1)
        b2 = np.roll(f_cos, -1)
        b1[0] = 30
        b2[-1] = (fs / 2) - 100

        self.freq_scale = fs * 1.0
        self.filt_b1 = fc
        self.N_filt = N_filt
        self.Filt_dim = Filt_dim
        self.fs = fs
        self.Batch_Size = Batch_Size
        #self.Q = Q

    def forward(self, QTemp):

        filters = Variable(torch.zeros((self.Batch_Size,self.N_filt, int((self.Filt_dim - 1))))).double().to(device)
        N = self.Filt_dim
        t_right = Variable(torch.linspace(1, (N - 1) , steps=int((N - 1))) / self.fs).to(device)
        alphaAll = Variable(torch.zeros((self.Batch_Size, self.N_filt))).double().to(device)

        filt_beg_freq = self.filt_b1/ self.freq_scale#(torch.abs(self.filt_b1) + min_freq / self.freq_scale).double().to(device)
        #filt_end_freq = filt_beg_freq + (torch.abs(self.filt_band) + min_band / self.freq_scale)
        Q = QTemp
        n = torch.linspace(0, N-1, steps=N-1)

        for i in range(self.N_filt):

            band_pass,alpha = gabor(filt_beg_freq[i] * self.freq_scale, t_right, self.fs, Q[:,i],self.Batch_Size)

            fft_bp = np.abs(np.fft.rfft(band_pass.transpose(1,0).detach().cpu(),n=512))
           # plt.plot(np.abs(fft_bp[2,:]),'b')

            maxfft = np.max(fft_bp)
            #plt.plot(i,maxfft,'*')

            band_pass = (band_pass / maxfft).double()
            alphaAll[:, i] = alpha

          #  fft_bp_norm = np.abs(np.fft.rfft(band_pass.transpose(1,0).detach().cpu(),n=512))
          #  plt.plot(np.abs(fft_bp_norm[3,:]),'r')
          #  temp = band_pass.transpose(1, 0).to(device)
          #  fftTemp = np.abs(np.fft.rfft(temp.detach().cpu()))
          #  plt.plot(np.abs(fftTemp[2, :]))

            filters[:, i, :] = (torch.flip(band_pass.transpose(1, 0), (1,)).to(device)).double() #Filter must be flipped to perform convolution. Otherwise, conv1d performs correlation

        return filters


class gabor_variableResponse(nn.Module):

    def __init__(self, N_filt, Filt_dim, Batch_Size, fs, fc):
        super(gabor_variableResponse, self).__init__()

        # Mel Initialization of the filterbanks
        low_freq_mel = 80
        high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, N_filt)  # Equally spaced in Mel scale

        fdelta = (fs / 2) / (N_filt + 1)
        f_cos = np.arange(46, (fs / 2) - 100, fdelta)  # np.arange(fdelta/2, fs/2, fdelta)

        b1 = f_cos
        b2 = np.roll(f_cos, -1)
        b1[0] = 30
        b2[-1] = (fs / 2) - 100

        self.freq_scale = 1.0 * 1.0
        self.filt_b1 = fc  # torch.from_numpy(b1 / self.freq_scale)#torch.from_numpy(b1 / self.freq_scale)#nn.Parameter(torch.from_numpy(b1 / self.freq_scale))

        self.N_filt = N_filt
        self.Filt_dim = Filt_dim
        self.fs = fs
        self.Batch_Size = Batch_Size

    def forward(self, QTemp):

        N = self.Filt_dim
        t_right = Variable(torch.linspace(1, (N - 1), steps=int((N - 1))) / self.fs).to(device)

        filt_beg_freq = self.filt_b1 / self.freq_scale  # (torch.abs(self.filt_b1) + min_freq / self.freq_scale).double().to(device)

        Q = QTemp
        n = torch.linspace(0, N - 1, steps=N - 1)

        BW = torch.div(filt_beg_freq, Q).double()
        # print(BW)
        alpha = BW / 2 * np.sqrt((2 * np.pi))
        b = (alpha / fs).double()
        Omega_c = (2 * math.pi * filt_beg_freq / fs).double()
        T = 1 / fs
        nT = (t_right * fs).double()
        n = nT.repeat([self.Batch_Size, filt_beg_freq.shape[1], 1])  # transpose(1, 0)
        y_right = (torch.exp(-(torch.pow(b.unsqueeze(2).expand_as(n), 2)) * torch.pow(n, 2))) * \
                  (torch.cos(Omega_c.unsqueeze(2).expand_as(n) * n))

        y = torch.flip(y_right, (2,))  # torch.cat([y_left.double(), Variable(torch.ones([1,batchSize], dtype=torch.double)).to(device), y_right.double()])

        return y


def gabor(band, t_right, fs, Q, batchSize):

    BW = (band / Q).double()
    #print(BW)
    alpha = BW/2 * np.sqrt((2 * np.pi))
    b = (alpha / fs).double()
    Omega_c = (2 * math.pi * band / fs).double()
    T = 1 / fs
    nT = (t_right * fs).double()
    n = nT.repeat([batchSize,1]).transpose(1,0)
    y_right = (torch.exp(-(torch.pow(b.expand_as(n), 2)) * torch.pow(n, 2))) * (torch.cos(Omega_c.expand_as(n) * n))

    y = y_right

    return y, alpha