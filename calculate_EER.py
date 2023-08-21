# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 23:52:04 2020

Author: Buddhi Wickramasinghe

Calculate EER from the network predictions

"""
import numpy as np
#import matplotlib.pyplot as plt
import scipy.io
from sklearn import metrics
import matplotlib.pyplot as plt

def calculate_EER(TePredOut):

    num_of_files = TePredOut.shape[0]*TePredOut[0].shape[0]
    TestLab_Ge1_Sp0=scipy.io.loadmat('data/labels_devSel.mat')
    TeLab_Ge1_Sp0Or=TestLab_Ge1_Sp0['labels_devSel']
    TeLab_Ge1_Sp0Or = np.squeeze(TeLab_Ge1_Sp0Or)
    TeLab_Ge1_Sp0Or = TeLab_Ge1_Sp0Or[0:num_of_files]

    Epsilon= 0.000001
    TePred = []
    for i in range(TePredOut.shape[0]):
        TePred.append(TePredOut[i].squeeze())

    TePred = np.vstack(TePred)
    TeLogRatio = TePred[:,1]-(TePred[:,0]) # (Gen Score/ Spoof Score) (Classes: 0,1. So probabilities are for 0, 1 in that order)
    fpr, tpr, thresholds = metrics.roc_curve( TeLab_Ge1_Sp0Or, TeLogRatio, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    EER1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    EER2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
    print('EER1:', EER1)
    print('EER2:', EER2)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) - Neural Net features')
    plt.show()
    return EER1