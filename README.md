# Tiny AI Environmental Monitoring System for TESSERACT data stream

This repository contains all codes being developed for a real-time anomalous noise detection system for the data stream from the HeRALD detector at Lawrence Berkeley National Laboratory, as part of the TESSERACT experiment. This branch contains workflow for hardware-general setup.

### Workflow

This architecture is designed for Zynq board-based hardware (Eclypse Z7, Pynq board, ...) connected to a single output channel from the HeRALD detector. Currently is generalized to any such hardware-- needs an updated ```acquire_into``` function in ```preprocess.py```.
Entry into workflow is ```main.sh```. The script begins ```preprocess.py``` and ```anomaly.py```.

### Preprocessing

Two 10 sec raw data buffers are stored on RAM at a time. One is stored while the other is preprocessed to ensure continuous streaming. Preprocessing includes signal peak removal around a radius and downsampling. After downsampling, the 10 sec chunk is saved as .npz file to a directory in microSD storage. The directory is pruned to fit a 120 chunk = 20 minute sliding window. Data outside this window is discarded.
Arguments are configured in ```config.json```.

### Modules

Modules are toggleable anomaly detection algorithms. Additional modules can be appended to the ```modules``` directory following the instructions in ```README.md``` in the directory. Default modules are:

- ```pca.py```: Uses PCA reconstruction error to evaluate anomalies. Fits PCA model on first half of sliding window, then transforms/reconstructs second half of sliding window. Evaluates anomalies in the reconstructed set.

- ```ema.py```: uses EMA baseline residuals to evaluate anomalies.

The modules are activated in ```anomaly.py```, which loads preprocessed data chunks on the microSD and runs the detection algorithms to create a binary anomaly flag array. The thresholding parameters are fixed in the current version but should be tuned periodically in a future version (potentailly against a benchmark e.g. inserted noise).

## Required Packages

- ```numpy```
- ```sklearn```
