import numpy as np
import os, time, glob, threading, queue, json, argparse
from preprocessing.utils import chunk, fft, preprocess

# ---------------------------------------------------------------------------
# Default configuration. An external config file overrides any of these keys;
# anything the file omits falls back to these values, so a partial or missing
# config still runs.
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    'output_dir': '/media/sd/processed',
    'downsample_factor': 10,
    'n_chunks': 1,
    'sampling_rate': 1.25e6,
    'channel_number': 0,
    'chunk_seconds': 10,
    'retention_seconds': 20 * 60,
}


def load_config(path=None):
    """
    Build the runtime config: start from DEFAULT_CONFIG, then overlay values
    from an external JSON file if one is given. Unknown keys in the file are
    kept (harmless), missing keys fall back to defaults.
    """
    cfg = dict(DEFAULT_CONFIG)          # copy so we never mutate the defaults
    if path:
        with open(path, 'r') as f:
            file_cfg = json.load(f)
        cfg.update(file_cfg)

    # Warn about unexpected keys so typos in the config don't fail silently.
    unknown = set(cfg) - set(DEFAULT_CONFIG)
    for k in unknown:
        print(f"[config] warning: unrecognized key '{k}' (ignored by script)")

    # sampling_rate may arrive as a string like "1.25e6" from JSON; coerce it.
    cfg['sampling_rate'] = float(cfg['sampling_rate'])
    cfg['downsample_factor'] = int(cfg['downsample_factor'])
    cfg['n_chunks'] = int(cfg['n_chunks'])
    cfg['channel_number'] = int(cfg['channel_number'])
    cfg['chunk_seconds'] = int(cfg['chunk_seconds'])
    cfg['retention_seconds'] = int(cfg['retention_seconds'])
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Acquire, preprocess, and save waveform chunks.")
    parser.add_argument('--config', type=str, default=None,
                        help='Path to a JSON config file. Missing keys fall back to defaults.')
    return parser.parse_args()


def acquire_into(buf):
    """
    HARDWARE-SPECIFIC: fill the preallocated array `buf` in place with one
    10-second chunk. Block until full. Return number of valid samples.

    Writing in place (rather than returning a new array) is what lets the
    ping-pong buffering reuse the same two arrays forever with no allocation.
    """
    raise NotImplementedError("Connect this to your acquisition path.")


def producer(bufs, full_q, free_q, stop_evt):
    """
    Acquisition thread. Continuously pulls an empty buffer from free_q,
    fills it via the hardware, and hands it to the consumer via full_q.
    This is one half of the ping-pong: it only ever writes to buffers the
    consumer isn't currently reading.
    """
    while not stop_evt.is_set():
        idx = free_q.get()              # block until a buffer is free to fill
        try:
            n = acquire_into(bufs[idx])  # fill that buffer in place
            full_q.put((idx, n))         # hand ownership to the consumer
        except Exception as e:
            print(f"[{time.time():.3f}] acquire error: {e}")
            free_q.put(idx)              # on error, return buffer so it isn't leaked


def preprocess_buffer(waveform_data, cfg):
    """
    Turn one raw waveform buffer into frequency/ASD arrays ready to save.
    Runs in the consumer thread. Heavy NumPy work here releases the GIL,
    letting the producer keep acquiring in parallel.
    """
    n_samples = waveform_data.shape[0]
    sr = cfg['sampling_rate']
    time_data = np.arange(n_samples) / sr          # time axis for the raw samples

    target_len = n_samples // cfg['downsample_factor']  # length after decimation
    duration = n_samples // sr                          # chunk duration in seconds
    fft_fs = target_len // duration                     # effective sample rate post-downsample

    # Downsample/clean the waveform, then split into n_chunks sub-segments.
    new_tdata, new_data = preprocess(time_data, waveform_data, target_len)
    chunks = chunk(new_tdata, new_data, cfg['n_chunks'])

    # FFT each sub-segment, collecting frequency bins and amplitude spectral density.
    # Cast to float32 to halve on-disk size versus float64.
    freqs_list, asd_list = [], []
    for c in chunks:
        f, a = fft(c[1], fft_fs)
        freqs_list.append(f.astype(np.float32))
        asd_list.append(a.astype(np.float32))
    return np.stack(freqs_list), np.stack(asd_list)


def save_chunk(freqs, asd, cfg):
    """
    Write one processed result to a timestamped .npz. The timestamp in the
    filename doubles as a sort key for pruning (chronological string sort).
    """
    ts = time.time()
    path = os.path.join(cfg['output_dir'], f"chunk_{ts:.3f}.npz")
    np.savez(path, freqs_list=freqs, asd_list=asd, timestamp=ts)
    return path


def prune_window(cfg, max_chunks):
    """Keep only the newest max_chunks files, deleting the oldest."""
    files = sorted(glob.glob(os.path.join(cfg['output_dir'], "chunk_*.npz")))
    while len(files) > max_chunks:
        try:
            os.remove(files.pop(0))
        except OSError:
            pass


def consumer(bufs, full_q, free_q, stop_evt, cfg, max_chunks):
    while not stop_evt.is_set():
        try:
            idx, n = full_q.get(timeout=1.0)
        except queue.Empty:
            continue
        try:
            data = bufs[idx][:n].copy()
        finally:
            free_q.put(idx)
        try:
            freqs, asd = preprocess_buffer(data, cfg)
            save_chunk(freqs, asd, cfg)
            prune_window(cfg, max_chunks)
        except Exception as e:
            print(f"[{time.time():.3f}] process error: {e}")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Derived quantities — computed after config load, no longer module constants.
    buf_samples = int(cfg['sampling_rate'] * cfg['chunk_seconds'])
    max_chunks = cfg['retention_seconds'] // cfg['chunk_seconds']

    print(f"[config] output_dir={cfg['output_dir']}")
    print(f"[config] sampling_rate={cfg['sampling_rate']:.3g} Hz, "
          f"chunk={cfg['chunk_seconds']}s -> {buf_samples} samples/buffer")
    print(f"[config] retention={cfg['retention_seconds']}s -> keep {max_chunks} files")

    os.makedirs(cfg['output_dir'], exist_ok=True)

    bufs = [np.empty(buf_samples, dtype=np.float32) for _ in range(2)]
    full_q, free_q = queue.Queue(), queue.Queue()
    for i in range(len(bufs)):
        free_q.put(i)

    stop_evt = threading.Event()
    p = threading.Thread(target=producer, args=(bufs, full_q, free_q, stop_evt))
    c = threading.Thread(target=consumer,
                         args=(bufs, full_q, free_q, stop_evt, cfg, max_chunks))
    p.start(); c.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_evt.set()
        p.join(); c.join()


if __name__ == '__main__':
    main()