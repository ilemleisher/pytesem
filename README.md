# Tiny AI Environmental Monitoring System for TESSERACT data stream

This repository contains all codes being developed for a real-time anomalous noise detection system for the data stream from the HeRALD detector at Lawrence Berkeley National Laboratory, as part of the TESSERACT experiment. The main branch is set up for direct implementation into the local PC used for data acquisition in the TESSERACT lab.

## Workflow

Entry into the workflow is ```main.sh```. Currently needs the arguments to be edited directly in the script. Future version will add support for command line specification.

```main.sh``` begins ```run_daq.py``` and ```run_naq.py```. Both scripts exchange PIDs to ensure they run and end together. ```run_daq.py``` is found in [pytesdaq](https://github.com/spice-herald/pytesdaq/blob/lbl_update/bin/run_daq.py) repo.

```run_naq.py``` calls functions from three packages: ```preprocessing```, ```data_reduction```, and ```analysis```. During the preprocessing block, data chunks that are saved locally into hdf5 files and loaded one at a time. Signal peaks are removed from each chunk, they are downsampled, and converted into power spectra. Each spectrum is stored in memory until a window of set length is full. Whenever a new chunk is added to the window, the oldest chunk is discarded. Each time this happens, the data reduction block is triggered. A PCA model is fit onto all but the newest chunk in the window. The newest chunk is then projected into PCA space and reconstructed back. Specific data points that highlight noise signatures are saved following this. These data points are given to the analysis block, which creates a live figure showing how these data points, as well as additionally calculated metrics of noise evaluation, evolve over time. This figure is saved when ```run_daq.py``` ends. The resulting saved outputs are the raw noise metrics following data reduction, and the complete visual evaluation metric history.

## Required Packages

- ```h5py```
- ```numpy```
- ```datetime```
- ```matplotlib```
- ```scipy```
- ```collections```
- ```sklearn```
