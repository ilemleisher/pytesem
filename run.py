import numpy as np
import sys, importlib
from dev.utils import get_files, filter_files, stitch_files

# Get path to modules
sys.path.append("/home/ilemleisher/em_project/modules")
# Load modules specified as command line arguments
modules = sys.argv[1:]
loaded_mods = {}
if len(sys.argv) < 2:
    print("No modules specified. Please provide module names as command line arguments.")

for name in sys.argv[1:]:
    try:
        loaded_mods[name] = importlib.import_module(name)
        print(f"Activated module: {name}")
    except Exception as e:
        print(f"Could not find module {name}: {e}")
    
# Path to preprocessed data (assumes format from preprocess.py)
path = "/home/ilemleisher/data/continuous_I4_D20250102_T224744/"
# Dataset
target = 'I4_D20250102_T224816'
print(f"Target: {target}")
# Load continuous data from preprocessed files following naming format from preprocess.py
filenames = filter_files(get_files(path),target)
# Stitch together continuous dataset
containers = stitch_files(path, filenames, 'freqs_list','asd_list')
freqs_total = containers['freqs_list']
asd_total = containers['asd_list']
# ASD to logspace
X = np.log10(asd_total + 1e-12).astype(np.float32)

print(f"Total: {len(X)} chunks")

# Create empty flags array to store anomaly labels for each chunk
flags = np.zeros(len(X), dtype=int)

# Run each module on the dataset
for name, mod in loaded_mods.items():
    print("------------------------------")
    print(f"Running module: {name}")
    flags, idx, metadata = mod.flag(freqs_total, X)
    if len(idx) > 0:
        flags += np.array(flags)
        print('ANOMALY DETECTED')
        print(f"Labeled chunks with indices {idx.tolist()} as anomalous by {name}.")
    else:
        print(f"NORMAL")
