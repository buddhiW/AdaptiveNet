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

### Data preparation

Speech utterances should first be decomposed into frames of length 11ms without overlapping. Each utterance must be stored in a separate _.npy_ file so that the              Dataloaders can read them. A set of sample frame speech files have been provided in data folder for training (_TrainSel.tar.gz_) and develpment (_DevSel.tar.gz_) partitions. 

### Running the experiment

To run the example replay detection experiment, execute the following command:

```
python main.py 
```


