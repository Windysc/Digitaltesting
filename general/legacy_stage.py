"""
legacy_stage.py -- run the ORIGINAL resampling function of the first data step
(Train_VAE_full/data_csv2npy.py, removed from the repository on 2026-09-24) on the
synthetic CSV files, so that check_mending_plan.py can compare the old chain with
the mended one.

`interpolation_downsample` below is that function, copied unchanged: a cubic
spline over the ROW INDEX of the track, evaluated at 100 equally spaced index
positions; no timestamp is read.  It needs scipy, which the pipeline itself does
not: run this file with an interpreter that has scipy and pandas.

    <python with scipy> legacy_stage.py check_out/csv check_out/legacy.npz
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import interpolate


def interpolation_downsample(data, num_points):
    """The original function of data_csv2npy.py, unchanged."""
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    elif len(data.shape) > 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {data.shape}")

    original_points = np.arange(len(data))
    new_points = np.linspace(0, len(data) - 1, num_points)

    if data.shape[1] == 1:
        tck = interpolate.splrep(original_points, data[:, 0], s=0)
        return interpolate.splev(new_points, tck)
    else:
        tck, _ = interpolate.splprep([data[:, i] for i in range(data.shape[1])], u=original_points, s=0)
        return np.column_stack(interpolate.splev(new_points, tck))


def main(csv_root, out_path):
    f = interpolation_downsample
    out = {}
    for root, _, files in os.walk(csv_root):
        for name in sorted(files):
            if not name.endswith('.csv'):
                continue
            df = pd.read_csv(os.path.join(root, name))
            lat_lon = df[['longitude_degrees', 'latitude_degrees']].values      # the original's column choice
            key = os.path.relpath(os.path.join(root, name), csv_root).replace(os.sep, '/')[:-4]
            try:
                out[key] = f(lat_lon, num_points=100)
            except Exception as e:                                               # keep going, record the failure
                out[key] = np.full((100, 2), np.nan)
                print('failed on %s: %s' % (key, e))
    np.savez(out_path, **out)
    print('legacy arrays: %d tracks -> %s' % (len(out), out_path))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
