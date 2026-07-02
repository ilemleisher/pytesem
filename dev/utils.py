import os, re, glob
import numpy as np
import time
from functools import wraps

def track_runtime(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"Function '{func.__name__}' took {end - start:.6f} seconds to execute.")
        return result
    return wrapper

def get_files(path='/data/lbl/run21/raw/continuous_I4_D20250102_T224744/'):
    """
    This function finds all files within a specified directory that follow tessy data naming trend and
    organizes it by continuity

    Parameters:
    - path: path to data directory

    Returns:
    - filenames: list of file names sorted by continuity
    """
    files = glob.glob(os.path.join(path, "*_I4_D*_T*_F*"))

    # Define pattern
    pat = re.compile(r"_I4_D(\d+)_T(\d+)_F(\d+)$")

    # Key function for sorting by F*
    def sort_key(fp):
        name = os.path.basename(fp)
        m = pat.match(name)
        if not m:
            return (10**18, 10**18, 10**18)
        d, t, f = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        return (d, t, f)
    
    filenames = [os.path.basename(fp) for fp in sorted(files, key=sort_key)]

    print(f"Found {len(filenames)} files in {path}")

    return filenames

def filter_files(filenames, target):
    """
    This function filters out filenames that don't match a specified target pattern

    Parameters:
    - filenames: list of filenames (e.g., from get_files())
    - target: target pattern to filter by

    Returns:
    - filtered_filenames: list of filenames that match the pattern
    """

    pat = re.compile(r"(I\d+_D\d+_T\d+)_F(\d+)(?:\.[^.]+)?$")

    out = []
    for f in filenames:
        m = pat.search(f)
        if m and m.group(1) == target:
            out.append((int(m.group(2)), f))

    out.sort(key=lambda x: x[0])

    filtered_filenames = [f for _, f in out]

    print(f"Filtered to {len(filtered_filenames)} files that match the target")

    return filtered_filenames

def stitch_files(path, filenames, *keys):
    """
    This function stitches together data files to make one continuous data array

    Parameters:
    - path: path to the files
    - filenames: consecutively sorted filenames (matches output of filter_files())
    - *keys: data key names in the files (e.g., 'asd_list', ...)

    Returns:
    - containers: a dictionary with keys corresponding to *keys parameters that contain the concatenated data arrays
    """
    if len(filenames) == 0:
        raise ValueError('No files found')
    
    containers = {}
    for key in keys:
        containers[key]=[]

    for filename in filenames:
        filepath = path+filename
        print("Reading:", filename)
        with np.load(filepath) as d:
            for key in keys:
                containers[key].append(d[key])

    for key in keys:
        containers[key] = np.concatenate(containers[key], axis=0)

    return containers
