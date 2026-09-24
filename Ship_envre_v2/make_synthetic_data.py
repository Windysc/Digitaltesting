"""
Stand-in for the generated AIS trace sets (the real dataset_1 / dataset_2 .npy
files are not distributed).  Produces two (n, 100, 2) [lon, lat] arrays in the
same layout as Train_VAE_full/data_csv2npy.py, so the environment code path is
identical when the real generated data are dropped in.

Scenario: own-ship guideline runs north-east for ~6 km.  The merge-line is a
roughly parallel trace, laterally offset and slightly curving away; the
environment walks the target BACKWARDS along it (target_reverse=True), which
yields a head-on / crossing approach that the attacker has to close.
"""
import os
import numpy as np

EARTH_R = 6371000.0


def _xy_to_lonlat(xy, lon0, lat0):
    lon = lon0 + np.degrees(xy[..., 0] / (EARTH_R * np.cos(np.radians(lat0))))
    lat = lat0 + np.degrees(xy[..., 1] / EARTH_R)
    return np.stack([lon, lat], axis=-1)


def make_synthetic_traces(n=200, T=100, seed=0, lon0=11.9, lat0=57.6,
                          own_speed=3.0, target_speed=3.5, trace_dt=20.0,
                          lateral_offset=800.0, noise_m=15.0):
    rng = np.random.default_rng(seed)
    t = np.arange(T) * trace_dt
    heading = np.radians(45.0)
    dir_ = np.array([np.cos(heading), np.sin(heading)])
    normal = np.array([-dir_[1], dir_[0]])

    guides, merges = [], []
    for k in range(n):
        # own guideline: straight-ish with a gentle random bend and sample-level jitter
        bend = rng.normal(0, 0.15)                      # rad over the full length
        s = own_speed * t
        ang = heading + bend * (s / s[-1])
        g = np.cumsum(np.stack([np.cos(ang), np.sin(ang)], 1) * own_speed * trace_dt, axis=0)
        g -= g[0]
        g += rng.normal(0, 150.0, 2) * normal            # lateral start scatter
        g += rng.normal(0, noise_m, g.shape)
        guides.append(g)

        # merge line: parallel lane, offset, curving away, random speed factor
        off = lateral_offset + rng.normal(0, 250.0)
        sp = target_speed * rng.uniform(0.8, 1.2)
        bend_m = rng.normal(0.25, 0.15)
        ang_m = heading + bend_m * (np.arange(T) / (T - 1))
        m = np.cumsum(np.stack([np.cos(ang_m), np.sin(ang_m)], 1) * sp * trace_dt, axis=0)
        m -= m[0]
        m += off * normal + dir_ * rng.uniform(500.0, 1500.0)
        m += rng.normal(0, noise_m, m.shape)
        merges.append(m)

    guides = _xy_to_lonlat(np.asarray(guides), lon0, lat0)
    merges = _xy_to_lonlat(np.asarray(merges), lon0, lat0)
    return guides, merges


def ensure_data(data_dir, n=200, seed=0):
    """Create data_dir/guideline.npy and mergeline.npy if missing; return the two paths."""
    os.makedirs(data_dir, exist_ok=True)
    gp = os.path.join(data_dir, 'guideline_synthetic.npy')
    mp = os.path.join(data_dir, 'mergeline_synthetic.npy')
    if not (os.path.exists(gp) and os.path.exists(mp)):
        g, m = make_synthetic_traces(n=n, seed=seed)
        np.save(gp, g)
        np.save(mp, m)
        print('synthetic traces written:', gp, mp, g.shape, m.shape)
    return gp, mp


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    gp, mp = ensure_data(os.path.join(here, 'data'))
    print(np.load(gp).shape, np.load(mp).shape)
