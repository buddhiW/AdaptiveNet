"""
Created on Wed Aug 12 2020

DNN Controlled Adaptive Front-end for Replay Attack Detection Systems

Author: Buddhi Wickramasinghe

* Contains parts of codes from https://github.com/mravanelli/SincNet.
Related publication: Ravanelli, Mirco, and Yoshua Bengio. "Speaker recognition from raw waveform with sincnet."
In 2018 IEEE spoken language technology workshop (SLT), pp. 1021-1028. IEEE, 2018.

"""

from __future__ import print_function
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from torch.autograd import Variable
import numpy as np
from numpy import matlib
from conf import *  # This will import all the variables in the cfg file
from filter_models import gabor_fixedResponse, gabor_variableResponse
import scipy.io
import matplotlib.pyplot as plt
import itertools
from read_data_labels import ASVSpoofDevData, ASVSpoofTrainData
import torch_dct as dct
from calculate_EER import calculate_EER
from torch.nn.init import kaiming_normal_, xavier_normal_
import os
from backend import *

class Filter_network(nn.Module):
    def __init__(self, batchSize, Qout_scale,
                 Qout_weights, Qout_biases, Qout_yshift, hidden_weights, hidden_biases, filtersFixed, FreqScale1,
                 fc_variableTiled, filter_mask, Qout_scaleFM, Qout_weightsFM, Qout_biasesFM, Qout_yshiftFM):
        super(Filter_network, self).__init__()

        self.bs = batchSize
        #self.padding1 = padding1
        #self.padding2 = padding2
        self.FreqScale1 = FreqScale1
        self.fc_variableTiled = fc_variableTiled
        self.filter_mask = filter_mask

        self.Qout_yshift = Qout_yshift
        self.Qout_scale = Qout_scale
        self.Qout_yshiftFM = Qout_yshiftFM
        self.Qout_scaleFM = Qout_scaleFM
        self.filtersFixed = filtersFixed

        ### Q calculation
        self.QoutLayerHidden = nn.Linear(cnn_N_filt[1],
                                         cnn_N_filt[1])  # Output from layer before classification+energy
        self.QoutLayerHidden2 = nn.Linear(cnn_N_filt[1],
                                         cnn_N_filt[1])
        self.QoutLayer = nn.Linear(cnn_N_filt[1],
                                   cnn_N_filt[1])
        self.bn1 = nn.BatchNorm1d(int(cnn_N_filt[1]))

        ### Creating filters
        self.gaborVar = gabor_variableResponse(cnn_N_filt[1], cnn_len_filt[1], batchSize, fs, fc_variable)
        self.gaborFixed = gabor_fixedResponse(cnn_N_filt[0], cnn_len_filt[0], batchSize, fs, fc)

        self.QoutLayer.weight = nn.Parameter(Qout_weights)
        self.QoutLayer.bias = nn.Parameter(Qout_biases)
        # self.deltaFactor = nn.Parameter(Qout_yshift)
        self.QoutLayerHidden.weight = nn.Parameter(hidden_weights)
        self.QoutLayerHidden.bias = nn.Parameter(hidden_biases)

        self.QoutLayerHidden2.weight = nn.Parameter(Qout_weightsFM)
        self.QoutLayerHidden2.bias = nn.Parameter(Qout_biasesFM)

        self.QoutLayer.weight.requires_grad = False
        self.QoutLayer.bias.requires_grad = False

        self.QoutLayerHidden.weight.requires_grad = True
        self.QoutLayerHidden.bias.requires_grad = True

        self.QoutLayerHidden2.weight.requires_grad = False
        self.QoutLayerHidden2.bias.requires_grad = False

    # Input (xIn): Framed speech utterance (num_frames x frame_size )
    # Each timestep: A frame
    def forward(self, xIn, qInitial, QPrev, prevFrame, prevEn):

        ### Generate filter coefficients
        filters_init = self.filtersFixed
        filters_variable = (self.gaborVar(QPrev)).reshape(self.bs * cnn_N_filt[1], cnn_len_filt[1] - 1)

        ### Filtering
        xInTemp = xIn.unsqueeze(1)  # F.pad(xIn.unsqueeze(0), (0, 30), mode='circular')

        filtout1 = F.conv1d(xInTemp, filters_init.view(cnn_N_filt[0], 1, cnn_len_filt[0] - 1), padding=0, groups=1)

        # Input shape: <minibatch_size, in_channels, input_length>
        filtoutT = filtout1.unsqueeze(0).view(self.bs, cnn_N_filt[0], -1)
        filtoutSD1 = filtoutT[:, 1:cnn_N_filt[0], :] - filtoutT[:, 0:cnn_N_filt[0]-1, :] ## Spatial differentiation
        filtoutSD2 = filtoutSD1  # filtoutSD1[:, 1:81, :] - filtoutSD1[:, 0:80, :] ## One-level spatial diff

        filtoutSD2Padded = torch.cat((prevFrame, filtoutSD2), 2)

        filtoutSD2T = filtoutSD2Padded.view(1, self.bs * cnn_N_filt[1], filtoutSD2Padded.shape[2])
        filtout = F.conv1d(filtoutSD2T, filters_variable.view(self.bs * cnn_N_filt[1], 1, cnn_len_filt[1] - 1),
                           padding=0,
                           groups=self.bs * cnn_N_filt[1])  # Filter shape: <out_channels, in_channels, filter_length>
        filtout = filtout.unsqueeze(0).view(self.bs, cnn_N_filt[1], -1)

        ### FM calculation
        fftFiltout = torch.abs(torch.fft.rfft(filtout, dim=2)) #For older PyTorch versions: complexAbs(torch.rfft(filtout, 1))
        fftFiltout = fftFiltout * self.filter_mask
        scf = torch.div(torch.sum(fftFiltout * self.FreqScale1, dim=2), torch.sum(fftFiltout, dim=2))
        fm = torch.abs(scf - self.fc_variableTiled)
        fm = self.bn1(fm)

        ### Calculate subband energy
        energy = torch.mean(torch.abs(filtoutSD2), 2)  # Energy of the first filterbank # Calculate energy of the frame and add extra dimension to concatenate
        dbEnergy = energy  # 20 * torch.log10(energy + 0.0000001)

        ### Q calculation
        Qin = dbEnergy  # torch.cat((avgFeature.squeeze(2), dbEnergy), dim=1)#torch.cat((temp, energy), dim=1)
        QhiddenMid = F.relu(self.QoutLayerHidden(fm))
        QhiddenMid = self.Qout_scaleFM * (F.tanh(self.QoutLayerHidden2(QhiddenMid))) + Qout_yshiftFM
        Qhidden = self.Qout_scale*(F.tanh(self.QoutLayer(Qin))) + Qout_yshift + QhiddenMid

        ### Compensation for edge effects
        prevFrame = filtoutSD2[:, :, filtoutSD2.shape[2] - cnn_len_filt[1] + 2:filtoutSD2.shape[2]]

        return filtout, prevFrame, Qhidden, fm


def complexAbs(x):
    real = x[:, :, :, 0]
    img = x[:, :, :, 1]
    absOut = torch.pow(torch.pow(real, 2) + torch.pow(img, 2) + 0.000000000001, 0.5)  # torch.pow returns nan if
    # the argument for a root is zero or negative
    return absOut


def normalization(x):
    meanTemp = x.mean(dim=2)
    stdTemp = x.std(dim=2) + 0.000001
    xNorm = (x - meanTemp.unsqueeze(2).repeat(1, 1, 251)) / stdTemp.unsqueeze(2).repeat(1, 1, 251)
    return xNorm


def deltas3(x, w):
    samples = x.shape[2]
    channels = x.shape[1]
    hlen = 1  # torch.floor(w/2)
    w = 2 * hlen + 1
    win = torch.range(-hlen, hlen, 1)  # flipped kernel#hlen:-1: -hlen;
    winR = win.repeat(channels, 1).double().to(device)
    xx = torch.cat([x[:, :, 0].unsqueeze(2), x, x[:, :, samples - 1].unsqueeze(2)], 2)
    d = F.conv1d(xx, winR.unsqueeze(1), padding=2, groups=channels)
    return d[:, :, 2:samples + hlen + 1]

def convOutputLength(input_length, kernel_size, stride, padding, dilation):
    outputLength = (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    return outputLength


def conv2DOutputLength(input_length, kernel_size, stride, padding, dilation):
    outputLengthH = (input_length[0] + 2 * padding - dilation * (kernel_size[0] - 1) - 1) / stride + 1
    outputLengthW = (input_length[1] + 2 * padding - dilation * (kernel_size[1] - 1) - 1) / stride + 1
    return outputLengthH, outputLengthW


def maxPoolOutputLength(input_length, kernel_size, stride, padding, dilation):
    outputLength = (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    return outputLength


def run_model(model1, model2, inputTensor, Qeq, QunitGain, batch_size, octFreqScale):
    num_frames = NUM_FRAMES  # np.shape(inputTensor)[0]
    num_batches = np.shape(inputTensor)[1]

    qout = torch.zeros((batch_size, cnn_N_filt[1], num_frames), device=device, requires_grad=False)
    fmout = torch.zeros((batch_size, cnn_N_filt[1], num_frames), device=device, requires_grad=False)
    prevFrame = torch.zeros((batch_size, cnn_N_filt[1], cnn_len_filt[1] - 2), device=device, dtype=torch.double)
    prevEn = torch.zeros((batch_size, cnn_N_filt[1]), device=device, dtype=torch.double)
    window = torch.hamming_window(wlen2).double().to(device)
    feature = torch.zeros((batch_size, cnn_N_filt[1]*5, num_frames + 1), device=device, dtype=torch.double,
                          requires_grad=False)
    filtered = torch.zeros((batch_size, cnn_N_filt[1], num_frames * wlen), device=device, dtype=torch.double,
                           requires_grad=False)  # Leaf nodes are user created variables. They do not have a grad function.
    varFilters = np.zeros((num_frames,), dtype=np.object)
    win = torch.zeros((3,), device=device, dtype=torch.double, requires_grad=False)
    win[0] = -1
    win[2] = 1
    qInitial = Qeq
    qIn = QunitGain

    # print(filtered.grad_fn.next_functions)
    # filtered.register_hook(print)

    for i in range(num_frames):

        a, prevFrame, qIn, prevEn = model1(inputTensor[i], qInitial,  qIn, prevFrame, prevEn)
        a.detach()
        filtered[:, :, i * wlen:wlen * (i + 1)] = a
        prevFrame.detach()
        qIn.detach()
        qout[:, :, i] = qIn
        fmout[:, :, i] = prevEn

    filtered = torch.abs(filtered)#torch.abs(torch.cat(filtered, 2))

    for i in range(cnn_N_filt[1]):
        filtStft = complexAbs(torch.stft(filtered[:, i, :], wlen2, shiftLen2, wlen2, window=window))
        k = i * 5
        feature[:, k, :] = torch.div(torch.sum(filtStft*octFreqScale[:,0,:,:],dim=1),
                                     torch.sum(octFreqScale[:,0,:,:],dim=1)) #torch.mean(filtStft, 1)  #

        feature[:, k+1, :] = torch.div(torch.sum(filtStft * octFreqScale[:,1,:,:], dim=1),
                                       torch.sum(octFreqScale[:,1,:,:], dim=1))  # torch.mean(filtStft, 1)

        feature[:, k+2, :] = torch.div(torch.sum(filtStft * octFreqScale[:,2,:,:], dim=1),
                                     torch.sum(octFreqScale[:,2,:,:], dim=1))  # torch.mean(filtStft, 1)

        feature[:, k+3, :] = torch.div(torch.sum(filtStft * octFreqScale[:,3,:,:], dim=1),
                                         torch.sum(octFreqScale[:,3,:,:], dim=1))  # torch.mean(filtStft, 1)  #

        feature[:, k+4, :] = torch.div(torch.sum(filtStft * octFreqScale[:,4,:,:], dim=1),
                                         torch.sum(octFreqScale[:,4,:,:], dim=1))  # torch.mean(filtStft, 1)  #


    featureT = feature
    feature = dct.dct(torch.log(feature.permute(0, 2, 1) + 0.00000001), norm='ortho')
    feature = feature.permute(0, 2, 1)
    feature = feature[:, 0:coeffs, :]
    dataDelta = deltas3(feature, 3)
    dataDelta2 = deltas3(dataDelta, 3)
    feature = torch.cat([feature, dataDelta.double(), dataDelta2.double()], dim=1)

    #feature = normalization(feature)

    output = model2(feature)

    return output, featureT


def load_checkpoint(filepath, model):
    checkpoint = torch.load(filepath)
    # model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model

def train(model1, model2, train_loader, optimizer):

    for i, (data, target) in enumerate(train_loader):

        data = Variable(data.to(device).contiguous())
        target = Variable(target.squeeze().to(device).contiguous())
        # with autograd.detect_anomaly():
        output, feature = run_model(model1, model2, data.permute(1, 0, 2), Qeq, QunitGain, FIXED_BATCH_SIZE,
                                    octFreqScale)
        loss = F.nll_loss(output, target)  # Mean loss over all samples
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def validate(model1, model2, data_loader):
    model1.eval()
    model2.eval()

    total_loss = 0
    correct = 0
    total = 0
    for i, (data, target) in enumerate(data_loader):

        data = Variable(data.to(device).contiguous())
        target = Variable(target.squeeze().to(device).contiguous())
        output, feature = run_model(model1, model2, data.permute(1, 0, 2), Qeq, QunitGain, FIXED_BATCH_SIZE,
                                    octFreqScale)

        ## Accuracy
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == target).sum().item()
        total += target.size(0)
        loss = F.nll_loss(output, target)
        total_loss = total_loss + loss.item()

    total_loss = total_loss / (len(data_loader))
    accuracy = 100 * correct / total
    return total_loss, accuracy

def compute_predictions(model1, model2, data_loader):
    num_batches = len(data_loader)  # int(snt_te / FIXED_BATCH_SIZE)
    allOutputs = np.zeros((num_batches,), dtype=np.object)
    allFeatures = np.zeros((num_batches,), dtype=np.object)

    model1.eval()
    model2.eval()

    for i, (data, target) in enumerate(data_loader):

        data = Variable(data.to(device).contiguous())
        target = Variable(target.squeeze().to(device).contiguous())
        output, feature = run_model(model1, model2, data.permute(1, 0, 2), Qeq, QunitGain, FIXED_BATCH_SIZE,
                                    octFreqScale)
        allOutputs[i] = output.data.cpu().numpy()

    return allOutputs
############################################################################

def main(args):

    model1 = Filter_network(FIXED_BATCH_SIZE, Qout_scale, Qout_weights, Qout_biases, Qout_yshift, hidden_weights,
                            hidden_biases, filter_fixed, FreqScale1, fc_variable, filter_mask, Qout_scaleFM, Qout_weightsFM,
                            Qout_biasesFM, Qout_yshiftFM).to(device)

    model1 = model1.double()
    model2 = ResNet().to(device).double()

    optimizer = optim.Adam(itertools.chain(model1.parameters(), model2.parameters()), lr=lr)

    best_loss = np.inf
    fileNum = 'all'
    checkpointsName = 'checkpoints' #+ str(fileNum)

    if args.resume:
        currCheckpoint = 5
        checkpoint = torch.load(checkpointsName + '/checkpoint_adapt_epoch_' + str(currCheckpoint) + '.pth')
        model1.load_state_dict(checkpoint['state_dict'])
    #
        checkpoint = torch.load(checkpointsName + '/checkpoint_backend_epoch_' + str(currCheckpoint) + '.pth')
        model2.load_state_dict(checkpoint['state_dict'])

        currCheckpoint = checkpoint['epoch']
    else:
        currCheckpoint = 0

    train_dataset = ASVSpoofTrainData()
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        FIXED_BATCH_SIZE, shuffle=True, drop_last=True)

    dev_dataset = ASVSpoofDevData()
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        FIXED_BATCH_SIZE, shuffle=False, drop_last=True)

    print('-' * 50)
    print('Adaptation-ASVSpoof2019: Energy+FM')
    print('-' * 50)
    start_time = time.time()

    if args.trainSystem is True:
        print('Training the system')

        model1.train()
        model2.train()

        for epoch in range(N_epochs):

            train(model1, model2, train_loader, optimizer)
            tr_loss, tr_acc = validate(model1, model2, train_loader)
            dev_loss, dev_acc = validate(model1, model2, dev_loader)

            print('Epoch: {} \tTrain loss: {:.6f}  Train acc: {:.6f}  |  Valid loss: {:.6f}  Valid acc: {:.6f}'.format(
                epoch, tr_loss, tr_acc, dev_loss, dev_acc))

            chkName1 = 'checkpoint_adapt_epoch_' + str(epoch+currCheckpoint+1) + '.pth'
            chkName2 = 'checkpoint_backend_epoch_' + str(epoch+currCheckpoint+1) + '.pth'

            checkpoint1 = {'epoch': epoch,
                           'state_dict': model1.state_dict(),
                           'optimizer': optimizer.state_dict(),
                           'dev_loss': dev_loss,
                           'dev_acc': dev_acc}

            checkpoint2 = {'epoch': epoch,
                           'state_dict': model2.state_dict(),
                           'optimizer': optimizer.state_dict(),
                           'dev_loss': dev_loss,
                           'dev_acc': dev_acc}

            if epoch % args.saveEpoch == 0:
                torch.save(checkpoint1, checkpointsName + '/' + chkName1)
                torch.save(checkpoint2, checkpointsName + '/' + chkName2)

            if dev_loss < best_loss:
                best_loss = dev_loss
                torch.save(checkpoint1, checkpointsName + '/' + 'best_model_adaptation.pth')
                torch.save(checkpoint2, checkpointsName + '/' + 'best_model_backend.pth')

        print("--- %s seconds ---" % (time.time() - start_time))

    if args.evalSystem:

        print('Computing EER')

        model1 = load_checkpoint(checkpointsName + '/checkpoint_adapt_epoch_3.pth', model1)
        model2 = load_checkpoint(checkpointsName + '/checkpoint_backend_epoch_3.pth', model2)
        preds = compute_predictions(model1, model2, dev_loader)
        calculate_EER(preds)
        exit()

if __name__ == "__main__":

    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.manual_seed(seed)
    np.random.seed(seed)

    parser = argparse.ArgumentParser(description='DNN-based replay detection using adaptive filterbanks')

    parser.add_argument('--trainSystem', default=True, type=bool,
                        help='Train the model')
    parser.add_argument('--evalSystem', default=True, type=bool,
                        help='Evaluate the system')
    parser.add_argument('--resume', default=False, type=bool,
                        help='Resume training from checkpoint')
    parser.add_argument('--saveEpoch', default=2, type=int,
                        help='Specify the epochs to save')

    args = parser.parse_args()

    #To use a different batch size for evaluation
    if args.evalSystem:
        FIXED_BATCH_SIZE = batch_size
    else:
        FIXED_BATCH_SIZE = batch_size

    NUM_FRAMES = num_frames - 1

    wlen = int(fs * cw_len / 1000.00)
    wlen2 = wlen * 2
    shiftLen2 = int(wlen2 / 2)
    num_classes = 2
    padding1 = 0
    padding2 = 0
    coeffs = 20

    mask = np.zeros((1, int((wlen2 / 2) + 1)))
    mask[0, 0:10] = 1
    res = fs / wlen2
    FreqScale = np.arange(0, wlen2) * res
    FreqScale = FreqScale[0:int(wlen2 / 2) + 1] * mask
    FreqScale = torch.Tensor(np.tile(np.expand_dims(FreqScale, 2), (FIXED_BATCH_SIZE, 1, NUM_FRAMES + 1))).double().to(
        device)

    mask = np.zeros((1, int((wlen2 / 2) + 1)))
    mask[0, 10:int((wlen2 / 2) + 1)] = 1
    res = fs / wlen2
    FreqScale2 = np.arange(0, wlen2) * res
    FreqScale2 = FreqScale2[0:int(wlen2 / 2) + 1] * mask
    FreqScale2 = torch.Tensor(np.tile(np.expand_dims(FreqScale2, 2), (FIXED_BATCH_SIZE, 1, NUM_FRAMES + 1))).double().to(
        device)

    res = fs / wlen
    FreqScale1 = np.arange(0, wlen) * res
    FreqScale1 = np.expand_dims(FreqScale1[0:int(wlen / 2) + 1], 0)
    FreqScale1 = torch.Tensor(np.tile(np.expand_dims(FreqScale1, 1), (FIXED_BATCH_SIZE, cnn_N_filt[1], 1))).double().to(
        device)

    ##### Loading pre-calculated values #####
    basefolder = 'data/Inputs/'

    ## Octave filterbank
    octFilts = scipy.io.loadmat(basefolder + 'octaveFiltbank.mat')
    octFreqScale = octFilts['octaveFiltbank']
    octFreqScale = torch.Tensor(np.tile(np.expand_dims(octFreqScale, 2), (FIXED_BATCH_SIZE, 1, 1, NUM_FRAMES + 1))).double().to(device)

    ## Import Q factors
    Q = scipy.io.loadmat(basefolder + 'Q_low_gabor_40.mat')
    Qmat = Q['Q_low_gabor_40']
    Qmat = np.expand_dims(Qmat[0, 0:cnn_N_filt[1]], axis=0)
    QunitGain = torch.Tensor(np.matlib.repmat(Qmat, FIXED_BATCH_SIZE, 1)).double().to(device)

    Q = scipy.io.loadmat(basefolder + 'Q_fixed_40.mat')
    Qmat = Q['Q_fixed_40']
    Qeq = torch.Tensor(np.matlib.repmat(Qmat, FIXED_BATCH_SIZE, 1)).double().to(device)

    ## Import centre frequencies
    f = scipy.io.loadmat(basefolder + 'fc_40.mat')
    fc = f['fc_40'].astype(np.float64)
    fc = torch.Tensor(fc).double().to(device)
    fc = fc.squeeze()

    f = scipy.io.loadmat(basefolder + 'fc_shifted_40.mat')
    fc_variableT = f['fc_shifted_40'].astype(np.float64)
    fc_variableT = fc_variableT[:, 0:cnn_N_filt[1]]
    fc_variable = torch.Tensor(np.matlib.repmat(fc_variableT, FIXED_BATCH_SIZE, 1)).double().to(device)

    ## Import filter mask
    m = scipy.io.loadmat(basefolder + 'sqMask_40.mat')
    filter_mask = m['sqMask_40']
    filter_mask = torch.Tensor(np.tile(filter_mask, (FIXED_BATCH_SIZE, 1, 1))).double().to(device)

    ## Deterministic Q layer
    coeffA = scipy.io.loadmat(basefolder + 'coeffA3_40.mat')
    Qout_scale = torch.squeeze(torch.Tensor(np.matlib.repmat(coeffA['coeffA3_40'], batch_size, 1)).double().to(device))
    Qout_scale = Qout_scale[:, 0:cnn_N_filt[1]]

    coeffB = scipy.io.loadmat(basefolder + 'coeffB3_40.mat')
    Qout_weights = torch.squeeze(torch.Tensor(coeffB['coeffB3_40']).double().to(device))
    Qout_weights = Qout_weights[0:cnn_N_filt[1], 0:cnn_N_filt[1]]

    coeffC = scipy.io.loadmat(basefolder + 'coeffC3_40.mat')
    Qout_biases = torch.squeeze(torch.Tensor(coeffC['coeffC3_40']).double().to(device))
    Qout_biases = Qout_biases[0:cnn_N_filt[1]]

    coeffD = scipy.io.loadmat(basefolder + 'coeffD3_40.mat')
    Qout_yshift = torch.squeeze(torch.Tensor(np.matlib.repmat(coeffD['coeffD3_40'], batch_size, 1)).double().to(device))
    Qout_yshift = Qout_yshift[:, 0:cnn_N_filt[1]]

    ## Clipping hidden layer output
    coeffA = scipy.io.loadmat(basefolder + 'coeffA_FM_40_2.mat')
    Qout_scaleFM = torch.squeeze(torch.Tensor(np.matlib.repmat(coeffA['coeffA_FM_40_2'], batch_size, 1)).double().to(device))
    Qout_scaleFM = Qout_scaleFM[:, 0:cnn_N_filt[1]]

    coeffB = scipy.io.loadmat(basefolder + 'coeffB_FM_40_2.mat')
    Qout_weightsFM = torch.squeeze(torch.Tensor(coeffB['coeffB_FM_40_2']).double().to(device))
    Qout_weightsFM = Qout_weightsFM[0:cnn_N_filt[1], 0:cnn_N_filt[1]]

    coeffC = scipy.io.loadmat(basefolder + 'coeffC_FM_40_2.mat')
    Qout_biasesFM = torch.squeeze(torch.Tensor(coeffC['coeffC_FM_40_2']).double().to(device))
    Qout_biasesFM = Qout_biasesFM[0:cnn_N_filt[1]]

    coeffD = scipy.io.loadmat(basefolder + 'coeffD_FM_40_2.mat')
    Qout_yshiftFM = torch.squeeze(torch.Tensor(np.matlib.repmat(coeffD['coeffD_FM_40_2'], batch_size, 1)).double().to(device))
    Qout_yshiftFM = Qout_yshiftFM[:, 0:cnn_N_filt[1]]

    ## Hidden Q control layer
    bias = np.ones((1,cnn_N_filt[1])) * np.spacing(1)
    hidden_biases = torch.squeeze(torch.Tensor(bias).double().to(device))
    weights = np.identity(cnn_N_filt[1])*0.005
    hidden_weights = torch.Tensor(weights).double().to(device)

    ## Layer 1 filterbank
    gaborFixed = gabor_fixedResponse(cnn_N_filt[0], cnn_len_filt[0], FIXED_BATCH_SIZE, fs, fc)
    filter_fixed = gaborFixed(Qeq)
    filter_fixed = filter_fixed[0, :, :]

    main(args)








