# AdaptiveNet

This project presents a system capable of detecting replay attacks on automatic speaker verification systems by dynamically recognizing and capturing environmental and device artifacts introduced during replay via filterbanks.

An example of replay attack detection using the Physical Attack partition of ASVspoof 2019 dataset (https://www.asvspoof.org/index2019.html) is provided. 

This code uses parts of codes from SincNet repository (https://github.com/mravanelli/SincNet) for utility functions such as parsing configuration files. 
Backend classification network is based on the models proposed in https://github.com/nesl/asvspoof2019.

## Requirements

* Python 3.6
* PyTorch 1.8.0
* torch-dct (https://github.com/zh217/torch-dct)
* pysoundfile (https://pysoundfile.readthedocs.io/en/latest/#)

## How to run a ASVspoof 2019 experiment

### Data preparation

Speech utterances should first be decomposed into frames of length 11ms without overlapping. Each framed utterance must be stored in a separate _.npy_ file so that the       Dataloaders can access them. A set of sample framed speech files have been provided in _data_ folder for training (_TrainSel.tar.gz_) and develpment (_DevSel.tar.gz_) partitions. 

For the code to run, _\[data\]_ section of the configuration file (_cfg/config_file.cfg_) should be modified according to the user paths. In the provided code, _tr_lst_ and _te_lst_ store the individual file paths of training and development set samples, respectively. _labTr_dict_ and _labTe_dict_ dictionaries assign the corresponding label to each sample.

Once the paths are set, run the following code to frame the signals:

```
python frame_signals.py
```
This is a one-time step for a given frame length.

### Running the experiment

To run the example replay detection experiment, execute the following command:

```
python main.py 
```


