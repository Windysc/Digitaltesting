"""
legacy_stage.py -- run the ORIGINAL resampling function of
Train_VAE_full/data_csv2npy.py on the synthetic CSV files.

The original file executes on import (it reads a hard-coded path), so the
function `interpolation_downsample` is taken from its source with `ast` and
executed unchanged.  It needs scipy, which the global Python lacks: run this
file with an interpreter that has scipy and pandas, e.g. the traffic venv.

    <python with scipy> legacy_stage.py check_out/csv check_out/legacy.npz
"""
import ast
import os
import sys

import numpy as np
import pandas as pd

# the original script: in the repository two levels up, else the Desktop checkout
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORIGINAL = next((p for p in (os.path.join(_REPO, 'Train_VAE_full', 'data_csv2npy.py'),
                             r'C:\Users\ASUS\Desktop\Digitaltesting-main\Digitaltesting-main\Train_VAE_full\data_csv2npy.py')
                 if os.path.isfile(p)), os.path.join(_REPO, 'Train_VAE_full', 'data_csv2npy.py'))


def original_function():
    src = open(ORIGINAL, encoding='utf-8').read()
    tree = ast.parse(src)
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'interpolation_downsample'][0]
    ns = {}
    exec('import numpy as np\nfrom scipy import interpolate\n' + ast.get_source_segment(src, fn), ns)
    return ns['interpolation_downsample']


def main(csv_root, out_path):
    f = original_function()
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
