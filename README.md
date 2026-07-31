# pytesem: Python Environmental Monitoring System for TESSERACT data stream

This repository is a real-time noise monitoring system for the TESSERACT experiment, that is designed to run on top of the [pytesdaq](https://github.com/spice-herald/pytesdaq) package. The main branch is set up for direct implementation into the local PC used for data acquisition in the TESSERACT lab.

## How to run

### Recommended: ```main.sh```

1. Edit ```main.sh``` directly to set the paths and parameters for your local setup, as well as arguments for ```run_daq.py```. 

2. Run:
    ```bash
    bash main.sh
```main.sh``` starts:

- ```run_daq.py``` (from [pytesdaq](https://github.com/spice-herald/pytesdaq))

- ```run_naq.py``` (this repository)

These processes exchange PIDs so they start/stop together.

### ```run_naq.py``` CLI arguments (as used by ```main.sh```)
    python run_naq.py \
    --data_dir <path_to_hdf5_input_folder> \
    --output_dir <path_to_output_folder> \
    --daq_pid <PID_of_running_run_daq.py_process> \
    --downsample_factor 10 \
    --sampling_rate 1.25e6 \
    --channel_number 0

Notes:

- ```data_dir``` is where ```run_daq.py``` writes waveform data to ```.hdf5``` files from the experimental output.
- ```output_dir``` is where noise features and figures are saved from ```run_naq.py```.

# Workflow

Entry into the workflow is ```main.sh```.

## Real-time processing pipeline (```run_naq.py```)

```run_naq.py``` orchestrates three internal packages: ```preprocessing```, ```feature_extraction```, and ```analysis```. 

### 1) Preprocessing

While new waveform data chunks are saved via ```run_daq.py``` to local ```.hdf5``` files, ```run_naq.py```:

1. Scans for finished new chunks (10 seconds each).
2. Loads chunks one at a time.
3. Removes signal peaks from the raw data.
4. Downsamples the waveform.
5. Converts the data to PSD.

The PSD for each chunk is held in memory until it contributes to a fixed-size sliding window.

### 2) Sliding window + feature extraction

```run_naq.py``` maintains a FIFO sliding window with a configurable number of chunks:

- When the window isn't full: chunks are accumulated.
- When the window is full: the addition of a new chunk discards the oldest chunk in the window. Feature extraction is performed on each window.

Feaure extraction is entirely based in Principal Component Analysis (PCA) reconstruction. 

### 3) PCA reconstruction residuals

With each new window, both short-term and long-term behaviors are analyzed:

- **Short-term**
    - Fit PCA model to all PSDs in the current window *except* from the newest chunk.
    - Use the fitted model to project the PSD from the newest chunk into PCA space, and then reconstruct it back into original sample space.
    - Compute the residual between the reconstructed PSD and the original.

- **Long-term**
    - Project/reconstruct the same final PSD in the window, but using PCA model fit to the first window in the run (stored in memory throughout ```run_naq.py```).
    - Compute the same residual.

These residuals (+ metadata) are saved to ```output_dir``` for each window.

### 4) Live analysis plots

The ```analysis``` block generates live updating figures that show how the residuals for the most anomalous frequency bins evolve over time. The final figure saves to ```output_dir``` at the end of the run.

## Output

All outputs are written to ```output_dir```. Window residuals + metadata are saved as ```.npz``` files like: ```output_dr/rd_<YYYYMMDDT%H%M%S>.npz```, where the timestamp corresponds to the newest chunk in that window. Each file contains arrays:

- ```residual```: short-term residual
- ```bins```: frequency axis for the stored bins
- ```asd```: the stored PSD
- ```long_term_residual```: long-term residual, only included for windows beyond the first

If at least one full window was processed, the live figure saved at the end of the run is ```output_dir/fig_final.png```.

## Required Packages

- ```h5py```
- ```numpy```
- ```datetime```
- ```matplotlib```
- ```scipy```
- ```collections```
- ```sklearn```
