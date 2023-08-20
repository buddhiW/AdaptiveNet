# Adaptive_DNN

This project presents a system capable of detecting replay attacks on automatic speaker verification systems by dynamically recognizing and capturing environmental and device artifacts introduced during replay via filterbanks.

An example of replay attack detection using the Physical Attack partition of ASVspoof 2019 dataset (https://www.asvspoof.org/index2019.html) is provided. 

This code uses parts of codes from SincNet repository (https://github.com/mravanelli/SincNet) for utility functions such as parsing configuration files. 
Backend classification network is based on the models proposed in https://github.com/nesl/asvspoof2019.

## Requirements

* Python 3.6
* PyTorch 1.8.0
* torch-dct (https://github.com/zh217/torch-dct)

## How to run a ASVspoof 2019 experiment

**1. Data preparation**
      
      Speech utterances should first be decomposed into frames of length 11ms without overlapping. Each utterance must be stored in a separate .npy file so that the                Dataloader can read them.   
2. 

