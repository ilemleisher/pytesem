# Tiny AI Environmental Monitoring System for TESSERACT data stream

This repository contains all codes being developed for a real-time anomalous noise detection system for the data stream from the HeRALD detector at Lawrence Berkeley National Laboratory, as part of the TESSERACT experiment. This branch uses a package layout that matches TESSERACT data taking software.

## Workflow

```main.sh``` is the entry into the workflow. Modify the arguments in it as desired. This script runs both ```preprocess.py``` and ```anomaly.py```. 

### Preprocessing

Waveform data from both ADCs in the detector are uploaded onto the TESSERACT servers in real-time in ~1 second increments. 
The data outputs are contained in HDF5 files. Current repository version reads data from past runs, not real-time.

- ```preprocess.py```: Loads all consecutive HDF5 files. Constructs waveform data from defined channel. Downsamples and chunks data. Filters out signal peaks and performs FFT. Concatenates chunks into one file. Saves as .npz to server.

### Modules

Modules are toggleable anomaly detection algorithms. Additional modules can be appended to the ```modules``` directory following the instructions in ```README.md``` in the directory. Default modules are:

- ```pca.py```: Uses PCA reconstruction error to evaluate anomalies.

- ```ema.py```: uses EMA baseline residuals to evaluate anomalies.

The modules are activated via ```anomaly.py```, which loads preprocessed data and runs the detection algorithms to create a binary anomaly flag array.

## Required Packages

- ```h5py```
- ```numpy```
- ```TensorFlow```
- ```matplotlib```
- ```sklearn```
