# pytesem: Python Environmental Monitoring System for TESSERACT data stream

This repository is a real-time noise monitoring system for the TESSERACT experiment, that is designed to run on top of the [pytesdaq](https://github.com/spice-herald/pytesdaq) package. The main branch is set up for direct implementation into the local PC used for data acquisition in the TESSERACT lab.

## How to run

### Recommended: ```run.sh```

### ```run.sh``` starts

- `run_daq.py` (from [pytesdaq](https://github.com/spice-herald/pytesdaq)) — acquires raw waveform data and writes it to timestamped `.hdf5` files in the output directory.
- `main.py` (this repository) — monitors the raw data directory in real time, downsamples/processes waveforms, and produces live noise-analysis figures.
- `post_run_analysis.py` (this repository) — runs after both processes finish, producing final summary output using the same threshold and frequency-cutoff settings.

`run_daq.py` and `main.py` exchange PIDs so they start and stop together, and `post_run_analysis.py` waits for both to complete before running.

1. **DAQ settings** (channels, duration, comment, etc.) for `run_daq.py` are hardcoded in `run.sh` and must be edited directly in the script if you want to change them. **This should be modified each run, generally**.

2. **Everything else** (run name, output directory, downsampling, sampling rate, frequency cutoff, channel number, thresholds, etc.) can be set via command-line flags when invoking `run.sh`, without editing the script. **Most of these do not need to be modified each run, generally**.

Run it with:

```bash
bash run.sh [options]
```

### ```run.sh``` command-line options

| Flag | Default | Description |
|---|---|---|
| `--run NAME` | `run45` | Run name; determines the raw data directory (`/home/mwilliams/data/<RUN>/raw`) and default output directory. |
| `--output-dir DIR` | `.../em_output/<RUN>/<NAME>` | Overrides the automatically generated output directory. Generally not recommended.|
| `--downsample-factor N` | `10` | Downsample reduction factor, passed to `main.py`. |
| `--sampling-rate HZ` | `1.25e6` | Sampling rate of the raw data (Hz), passed to `main.py`. |
| `--freq-cutoff HZ` | `1000.0` | Frequency (Hz) separating low/high bands; passed to both `main.py` and `post_run_analysis.py`. |
| `--channel-number N` | `0` | Channel number (0-indexed) in the raw data files, passed to `main.py`. |
| `--max-bins N` | `5` | Maximum number of bins shown in each `main.py` live figure panel. |
| `--threshold-low VAL` | `2.0` | Threshold used for low frequency bin selection; passed to both `main.py` and `post_run_analysis.py`. |
| `--threshold-high VAL` | `3.0` | Threshold used for high frequency bin selection; passed to both `main.py` and `post_run_analysis.py`. |
| `--band-width VAL` | `2.0` | Band width in live figures (the red zone begins at `threshold + 2 * band_width`). |
| `--figures LIST` | `worst_bins psd` | Which live figures to generate: worst_bins, psd, or both; passed to `main.py` |

In general, most of the defaults for these parameters are fine and do not need to be configured each run (with the exception of --run). So usage is simple:

```bash
bash run.sh --run run46 
```

Configuring the ```run_daq.py```settings is necessary each run though, and as said above, this must be done in the ```run.sh``` file itself. 

## ```main.py```

### 1) Preprocessing

While new waveform data chunks are saved via ```run_daq.py``` to local ```.hdf5``` files, ```main.py```:

1. Scans for finished new chunks (10 seconds each).
2. Loads chunks one at a time.
3. Removes signal peaks from the raw data.
4. Downsamples the waveform.
5. Converts the data to PSD.

The PSD for each chunk is held in memory until it contributes to a fixed-size sliding window.

### 2) Sliding window + feature extraction

```main.py``` maintains a FIFO sliding window with a configurable number of chunks:

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
    - Project/reconstruct the same final PSD in the window, but using PCA model fit to the first window in the run (stored in memory throughout ```main.py```).
    - Compute the same residual.

These residuals (+ metadata) are saved to ```output_dir``` for each window.

### 4) Live analysis plots

The ```analysis``` block generates live updating figures that show how the residuals for the most anomalous frequency bins evolve over time. The final figure saves to ```output_dir``` at the end of the run.

### Output

All outputs are written to ```output_dir```. Window residuals + metadata are saved as ```.npz``` files like: ```output_dr/rd_<YYYYMMDDT%H%M%S>.npz```, where the timestamp corresponds to the newest chunk in that window. Each file contains arrays:

- ```residual```: short-term residual
- ```bins```: frequency axis for the stored bins
- ```asd```: the stored PSD
- ```long_term_residual```: long-term residual, only included for windows beyond the first

If at least one full window was processed, the live figure saved at the end of the run is ```output_dir/fig_final.png```.

## ```post_run_analysis.py```

Loads reduced-data files (rd_*.npz) ```main.py``` saved during acquisition, reassembles them into chronological time series short-term long-term PCA residuals, scans that history flag any bin whose residual exceeded "red zone" threshold (defined threshold + 2 * band_width) either low- or high-frequency band. prints out sorted list every such crossing, with timestamp, term (short/long), frequency band, specific bin/frequency, size residual, so you can spot problems occurred during run.

## Required Packages

- ```h5py```
- ```numpy```
- ```datetime```
- ```matplotlib```
- ```scipy```
- ```collections```
- ```sklearn```
