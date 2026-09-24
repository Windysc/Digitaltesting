"""Probe the Evaluation/jsd_matrix.ipynb metric: does it separate good from bad traces?"""
import json, numpy as np
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
# the notebook: in the repository (../../Evaluation), else the Desktop checkout
_NB = next((p for p in (os.path.join(os.path.dirname(os.path.dirname(_HERE)), 'Evaluation', 'jsd_matrix.ipynb'),
                        r'C:\Users\ASUS\Desktop\Digitaltesting-main\Digitaltesting-main\Evaluation\jsd_matrix.ipynb')
            if os.path.isfile(p)), 'jsd_matrix.ipynb')
nb = json.load(open(_NB, encoding='utf-8'))
src = '\n'.join(''.join(c['source']) for c in nb['cells'][1:5])
src = src.replace('__name__', '"skip"').replace('__', 'U_')
g = {}
exec(src, g)
from scipy import stats
np.random.seed(0)

def base_trace(n=100):
    # 6 km straight-ish track near 57.6N, 11.9E, in degrees
    t = np.linspace(0, 1, n)
    lon = 11.9 + 0.08 * t + 0.002 * np.sin(3 * t)
    lat = 57.6 + 0.03 * t
    return np.column_stack([lon, lat])

def run(tr):
    seg = g['calculate_segment_distances'](tr)
    tot = g['calculate_total_distance'](tr)
    th = {"cdf": lambda p, x: stats.norm.cdf(x, p[0], abs(p[1]) + 1e-300),
          "likelihood": lambda p, d: -np.sum(stats.norm.logpdf(d, p[0], abs(p[1]) + 1e-300)),
          "params": [seg.mean(), seg.std()]}
    bs = {"iterations": 0, "percentiles": [2.5, 97.5], "blockSize": 1}
    r = g['TrajectoryJSD'](tr, {"start": tot - 1e-10, "stop": tot + 1e-10, "bins": 3},
                           {"start": seg.min(), "stop": seg.max(), "bins": 100}, th, bs)
    return r["JSD-TD"]["jsdEstimate"], r["JSD-SD"]["jsdEstimate"]

b = base_trace()
cases = {
    'clean real-like trace': b,
    'noise var 1e-8 (11 m)': b + np.random.normal(0, 1e-4, b.shape),
    'noise var 1e-4 (1.1 km, data_csv2npy)': b + np.random.normal(0, 1e-2, b.shape),
    'pure random walk (not a ship)': np.cumsum(np.random.normal(0, 1e-3, b.shape), 0) + b[0],
    'reversed trace': b[::-1],
    'trace moved 1 deg away': b + 1.0,
    'uniform random points in box': np.column_stack([np.random.uniform(11.9, 12.0, 100), np.random.uniform(57.6, 57.65, 100)]),
}
print(f"{'case':42s} JSD-TD   JSD-SD")
for k, tr in cases.items():
    td, sd = run(tr)
    print(f"{k:42s} {td:.4f}  {sd:.4f}")
