# Tiny AI Environmental Monitoring System for TESSERACT data stream

This repository contains all codes being developed for a real-time anomalous noise detection system for the data stream from the HeRALD detector at Lawrence Berkeley National Laboratory, as part of the TESSERACT experiment.

## Workflow

### Preprocessing

Waveform data from both ADCs in the detector are uploaded onto the TESSERACT servers in real-time in ~1 second increments. 
The data outputs are contained in HDF5 files. Current repository version reads data from past runs, not real-time.

- ```preprocess.py```: Loads all consecutive HDF5 files. Constructs waveform data from defined channel. Downsamples and chunks data. Filters out signal peaks and performs FFT. Concatenates chunks into one file. Saves as .npz to server.

### Modules

Modules are toggleable anomaly detection algorithms. Additional modules can be appended to the ```modules``` directory following the instructions in ```README.md``` in the directory. Default modules are:

- ```pca.py```: Uses PCA reconstruction error to evaluate anomalies.

- ```ema.py```: uses EMA baseline residuals to evaluate anomalies.

The modules are activated via ```main.py```, which loads preprocessed data and runs the detection algorithms to create a binary anomaly flag array.

### Development Scripts

Scripts in ```dev``` directory are used for analysis/development of the workflow. The only script in this directory that is required to run the complete workflow is ```utils.py```.

- ```preprocess.py```: reads the raw HDF5 files, downsamples them, divides each data file into uniform chunks, filters out any chunks that contain signal events (represented by peaks in the waveform data), performs FFT on the chunks, then concatenates the chunks so that they represent time-continuous data (across several data files).

- ```anomaly.py```: loads ```.npz``` files saved from ```preprocess.py``` and uses PCA to calculate a reconstruction error and identify data chunks that contain anomalous peaks. Also calculates EMA baseline residual to identify data chunks that contian anomalous baseline drift. Plots the PCA reconstruction residual against the ASD for each anomalous chunk, and plots 5 data chunks centered on each chunk labeled with a baseline anomaly.

- ```pca_tuning.py```: same structure as ```anomaly.py``` except there is parameter tuning to find optimal parameters for the PCA anomaly detection.

- ```model.py```: TensorFlow 2D Convolutional Neural Network (CNN). Has three 2D convolution layers, a global pooling layer, and a linear output layer.

- ```train.py```: loads data files saved from ```preprocess.py``` and corresponding label files from ```pca.py``` to create data patches for the CNN. Trains the model using binary cross entropy loss, saves the best model, metrics, and loss/accuracy plots.

- ```utils.py```: contains helper functions for loading files.

- ```infer.py```: work in progress, eventually to be used with the trained CNN.

- ```noise.py```: works the same as ```preprocess.py``` for loading in data and creating spectral chunks, however with the addition of artifical noise modes, with the goal being to implement specific noise to have a sufficiently large labeled dataset for supervised training of the CNN.

## Required Packages

- ```h5py```
- ```numpy```
- ```TensorFlow```
- ```matplotlib```
- ```sklearn```
