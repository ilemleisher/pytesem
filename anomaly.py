import numpy as np
import sys, importlib, argparse
from dev.utils import get_files, filter_files, stitch_files

def parse_args():
    parser = argparse.ArgumentParser(description="Run anomaly detection on preprocessed data.")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the folder containing the preprocessed .npz files.')
    parser.add_argument('--target', type=str, required=True, help='Target dataset name to filter files.')
    parser.add_argument('modules', nargs='+', help='List of anomaly detection modules to run.')
    return parser.parse_args()

def load_modules(module_names, package=".modules"):
    loaded_mods = {}
    for name in module_names:
        fqmn = f"{package}.{name}"
        try:
            loaded_mods[name] = importlib.import_module(fqmn)
            print(f"Activated module: {fqmn}")
        except ModuleNotFoundError as e:
            print(f"Could not find module {fqmn}: {e}")
        except Exception as e:
            print(f"Failed to load {fqmn}: {e}")
    return loaded_mods

def main():

    args = parse_args()

    path = args.input_dir
    target = args.target
    print(f"Target: {target}")

    loaded_mods = load_modules(args.modules)

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

if '__name__' == '__main__':
    main()