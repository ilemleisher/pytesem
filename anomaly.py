import numpy as np
import importlib, importlib.util, argparse, os, sys, time, glob, zipfile

def parse_args():
    parser = argparse.ArgumentParser(description="Run anomaly detection on preprocessed data.")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Folder containing the preprocessed .npz files.')
    parser.add_argument('--module_dir', type=str, default=None,
                        help='Folder on the microSD holding detection module .py files. '
                             'If omitted, modules are imported from the modules package.')
    parser.add_argument('--poll_interval', type=float, default=2.0,
                        help='Seconds between directory scans for new files.')
    parser.add_argument('--window', type=int, default=None,
                        help='Only run detection on the most recent N chunks (default: all).')
    parser.add_argument('modules', nargs='+', help='Detection module names to run.')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------
def load_module_from_path(name, module_dir):
    """Load a module by loading name.py directly from module_dir on the SD card."""
    path = os.path.join(module_dir, f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise ModuleNotFoundError(f"No spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_modules(module_names, module_dir=None, package="modules"):
    """
    Load each named detection module. If module_dir is given, load the .py
    file directly from that folder (easy to drop new detectors on the SD card).
    Otherwise fall back to importing from a package.
    """
    loaded_mods = {}
    for name in module_names:
        try:
            if module_dir:
                loaded_mods[name] = load_module_from_path(name, module_dir)
                print(f"Activated module: {name} (from {module_dir})")
            else:
                fqmn = f"{package}.{name}"
                loaded_mods[name] = importlib.import_module(fqmn)
                print(f"Activated module: {fqmn}")
        except FileNotFoundError as e:
            print(f"Could not find module file for {name}: {e}")
        except ModuleNotFoundError as e:
            print(f"Could not find module {name}: {e}")
        except Exception as e:
            print(f"Failed to load {name}: {e}")
    return loaded_mods


# ---------------------------------------------------------------------------
# File-safety guards
# ---------------------------------------------------------------------------
def is_stable(filepath, checks=2, interval=0.3):
    """
    Return True only if the file's size stays constant across `checks`
    consecutive samples. Catches files still being written by a non-atomic
    writer (e.g. rsync). Redundant with the producer's atomic os.replace,
    but harmless and covers files arriving by other means.
    """
    try:
        last = os.path.getsize(filepath)
    except OSError:
        return False
    for _ in range(checks - 1):
        time.sleep(interval)
        try:
            size = os.path.getsize(filepath)
        except OSError:
            return False
        if size != last:
            return False
        last = size
    return True


def safe_load_npz(filepath):
    """
    Try to open an .npz; return the loaded object or None if it's unreadable
    (truncated / corrupt / mid-write). np.load is lazy, so touching .files
    forces the zip central directory to actually be read and validated.
    """
    try:
        npz = np.load(filepath, allow_pickle=False)
        _ = npz.files
        return npz
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as e:
        print(f"Skipping unreadable file {os.path.basename(filepath)}: {e}")
        return None


# ---------------------------------------------------------------------------
# Dataset loading + detection
# ---------------------------------------------------------------------------
def load_dataset(path, window=None):
    """
    Load and stitch the most recent .npz files into continuous arrays, with no
    dependency on dev.utils. Grabs all chunk_*.npz files and keeps the newest
    `window` of them (or all if window is None).
    """
    filenames = sorted(glob.glob(os.path.join(path, "chunk_*.npz")))

    if window is not None:
        filenames = filenames[-window:]        # most recent N files only

    freqs_parts, asd_parts = [], []
    for fp in filenames:
        npz = safe_load_npz(fp)                 # skip unreadable / mid-write files
        if npz is None:
            continue
        try:
            freqs_parts.append(npz['freqs_list'])
            asd_parts.append(npz['asd_list'])
        except KeyError as e:
            print(f"Skipping {os.path.basename(fp)}: missing key {e}")

    if not asd_parts:
        return None, None

    # Concatenate all chunks along the first axis.
    freqs_total = np.concatenate(freqs_parts, axis=0)
    asd_total = np.concatenate(asd_parts, axis=0)

    # ASD to logspace (guard against log10(0)).
    X = np.log10(asd_total + 1e-12).astype(np.float32)
    return freqs_total, X


def run_detection(loaded_mods, freqs_total, X):
    """Run every loaded module on the dataset and accumulate per-chunk flags."""
    flags = np.zeros(len(X), dtype=int)        # one flag slot per chunk
    for name, mod in loaded_mods.items():
        print("------------------------------")
        print(f"Running module: {name}")
        try:
            mod_flags, idx, metadata = mod.flag(freqs_total, X)
        except Exception as e:
            print(f"Module {name} failed: {e}")
            continue
        idx = np.asarray(idx)
        if idx.size > 0:
            flags[idx] += 1                    # mark the chunks this module flagged
            print('ANOMALY DETECTED')
            print(f"Labeled chunks {idx.tolist()} as anomalous by {name}.")
        else:
            print("NORMAL")
    return flags


# ---------------------------------------------------------------------------
# Main watch loop
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    path = args.input_dir

    loaded_mods = load_modules(args.modules, module_dir=args.module_dir)
    if not loaded_mods:
        print("No modules loaded; nothing to run.")
        return

    print(f"Watching {path} for new files (poll every {args.poll_interval}s)...")
    seen = set()

    try:
        while True:
            current = set(os.path.basename(p) for p in glob.glob(os.path.join(path, "chunk_*.npz")))
            candidates = current - seen

            # Only accept files that have finished being written.
            ready = {f for f in candidates
                     if is_stable(os.path.join(path, f))}

            if ready:
                print("==============================")
                print(f"{len(ready)} new file(s) ready: {sorted(ready)}")

                freqs_total, X = load_dataset(path, window=args.window)
                if X is None:
                    print("No data to process.")
                else:
                    print(f"Total: {len(X)} chunks")
                    flags = run_detection(loaded_mods, freqs_total, X)
                    n_flagged = int(np.count_nonzero(flags))
                    print(f"Summary: {n_flagged}/{len(flags)} chunks flagged.")

                # Mark only ready files as seen; unstable ones get re-checked
                # next poll once they've settled — they're deferred, not lost.
                seen |= ready

            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        print("\nStopping watcher.")


if __name__ == '__main__':
    main()