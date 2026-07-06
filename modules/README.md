# Modules

Modules are called by ```run.py``` to flag anomalies in the data. Each module contains a unique flagging method. All modules must contain a ```flag()``` function that has at least two parameters: 

- ```X```: An array that contains ASD amplitudes in logspace for all data chunks. Has shape (N, F), where N is the number of data chunks and F is the number of frequency bins.

- ```freqs```: An array that contains ASD frequency bin values for all data chunks. Has shape (N, F), where N is the number of data chunks and F is the number of frequency bins.

Any additional parameters must have default values. An example of an additional parameter is a sensitivity knob (e.g., threshold value, sigma, etc). All ```flag()``` functions must return:

- ```flags```: An array of anomaly labels for each data chunk (0 = no anomaly, 1 = anomaly).

- ```idx```: An array of indices corresponding to anomalous data chunks.

- ```metadata```: A dictionary storing additional information that can be used for analysis or bug fixing.