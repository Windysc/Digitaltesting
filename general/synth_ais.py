"""
synth_ais.py -- synthetic AIS-like voyages with a known truth, for testing the
sampling and smoothing chain while the real data stay private.

Truth: a ship integrated at 1 s with speed changes (up to 0.05 m/s^2) and turns
(0.15-0.5 deg/s, turn rate building up with a 20 s lag).  Course is in degrees
clockwise from north, x east, y north.

Reports follow the Class A schedule of ITU-R M.1371 (10 s under 14 kn, 3.3 s
while changing course, 6 s at 14-23 kn, 2 s above or when turning fast), with
message loss, reception gaps, a position error made of a slowly varying part
(Gauss-Markov, the way GNSS errors behave) plus a white part, timestamps
rounded to the second with a small latency, SOG in 0.1 kn and COG in 0.1 deg.
Regimes: 'raw' (as received), 'ds30' / 'ds60' (provider keeps the first report
of every 30 s / 60 s).  'dirty' adds outliers and duplicated rows.
"""
import math
import os

import numpy as np

from ais_prep import KN, xy_to_ll

EPOCH0 = 1_740_000_000.0          # arbitrary absolute start, whole seconds


def make_truth(rng, duration_s, turns=None, rate_range=(0.15, 0.5)):
    n = int(duration_s)
    r_cmd, a_cmd = np.zeros(n), np.zeros(n)
    for _ in range(rng.integers(0, 4) if turns is None else turns):
        rate = rng.uniform(*rate_range) * rng.choice([-1.0, 1.0])
        ang = rng.uniform(15.0, 110.0)
        t0 = int(rng.uniform(0.08, 0.85) * n)
        r_cmd[t0:min(n, t0 + int(ang / abs(rate)))] += rate
    r_cmd = np.clip(r_cmd, -1.2 * rate_range[1], 1.2 * rate_range[1])
    for _ in range(rng.integers(0, 3)):
        acc = rng.uniform(0.02, 0.05) * rng.choice([-1.0, 1.0])
        t0 = int(rng.uniform(0.05, 0.9) * n)
        a_cmd[t0:min(n, t0 + int(rng.uniform(0.5, 2.0) / abs(acc)))] += acc
    rate = np.zeros(n)
    for k in range(1, n):
        rate[k] = rate[k - 1] + (r_cmd[k] - rate[k - 1]) / 20.0
    speed = np.clip(rng.uniform(3.5, 9.0) + np.cumsum(a_cmd), 2.0, 11.0)
    course = rng.uniform(0.0, 360.0) + np.cumsum(rate)
    c = np.radians(course)
    x = np.concatenate([[0.0], np.cumsum(speed * np.sin(c))[:-1]])
    y = np.concatenate([[0.0], np.cumsum(speed * np.cos(c))[:-1]])
    return dict(t=np.arange(n, dtype=float), x=x, y=y, speed=speed, course=course, rate=rate)


def truth_at(tr, t):
    """Truth positions, speed, course [deg] and turn rate [deg/s] at times t (seconds from voyage start)."""
    t = np.asarray(t, float)
    f = lambda a: np.interp(t, tr['t'], a)
    return np.stack([f(tr['x']), f(tr['y'])], axis=-1), f(tr['speed']), f(tr['course']), f(tr['rate'])


def sample_reports(rng, tr, regime='raw', loss=0.2, gaps=None, sigma_c=3.0, tau_c=100.0, sigma_w=2.0,
                   dirty=False, white_only=None, latency_max=1.2, runs=0, nan_sog=0.0, sentinels=0.0):
    n = len(tr['t'])
    t, out = 0.0, []
    while True:
        k = int(t)
        kn = tr['speed'][k] / KN
        turning = abs(tr['rate'][k]) > 0.05
        base = (10.0 / 3.0 if turning else 10.0) if kn < 14 else (2.0 if turning else 6.0) if kn < 23 else 2.0
        t += base + rng.uniform(-0.3, 0.3)
        if t >= n - 2:
            break
        if rng.random() > loss:
            out.append(t)
    t = np.array(out)
    if regime in ('ds30', 'ds60'):
        bucket = 30.0 if regime == 'ds30' else 60.0
        phase = rng.uniform(0, bucket)
        _, first = np.unique(np.floor((t + phase) / bucket), return_index=True)
        t = t[first]
    if gaps is None:
        gaps = [(rng.uniform(0.05, 0.9) * n, rng.uniform(60.0, 600.0)) for _ in range(rng.integers(0, 4))]
    for g0, g in gaps:
        t = t[(t < g0) | (t > g0 + g)]
    xy, sp, co, _ = truth_at(tr, t)
    if white_only is not None:
        err = rng.normal(0.0, white_only, (len(t), 2))
    else:
        rho = math.exp(-1.0 / tau_c)
        ou = np.zeros((n, 2))
        e = rng.normal(0.0, sigma_c * math.sqrt(1 - rho * rho), (n, 2))
        ou[0] = rng.normal(0.0, sigma_c, 2)
        for k in range(1, n):
            ou[k] = rho * ou[k - 1] + e[k]
        err = ou[t.astype(int)] + rng.normal(0.0, sigma_w, (len(t), 2))
    pos = xy + err
    is_out = np.zeros(len(t), bool)
    if dirty:
        hit = rng.random(len(t)) < 0.004
        ang = rng.uniform(0, 2 * math.pi, len(t))
        mag = rng.uniform(150.0, 3000.0, len(t))
        pos[hit] += np.stack([mag * np.cos(ang), mag * np.sin(ang)], axis=1)[hit]
        is_out |= hit
        if len(t) > 50:                                     # one run of two bad fixes with the same offset
            j = int(rng.integers(20, len(t) - 20))
            pos[j:j + 2] += np.array([600.0, -500.0])
            is_out[j:j + 2] = True
    for _ in range(runs):                                   # a run of 3 to 10 fixes with one lasting offset
        if len(t) > 80:
            j, m = int(rng.integers(20, len(t) - 30)), int(rng.integers(3, 11))
            ang = rng.uniform(0, 2 * math.pi)
            pos[j:j + m] += rng.uniform(300.0, 800.0) * np.array([math.cos(ang), math.sin(ang)])
            is_out[j:j + m] = True
    t_rep = np.floor(EPOCH0 + t + rng.uniform(0.0, latency_max, len(t)))
    sog = np.round((sp + rng.normal(0.0, 0.05, len(t))) / KN, 1)
    cog = np.round((co + rng.normal(0.0, 0.6, len(t))) % 360.0, 1)
    sog[rng.random(len(t)) < nan_sog] = np.nan              # receiver without a speed for some reports
    miss = rng.random(len(t)) < sentinels                   # AIS 'not available': 102.3 kn and 360.0 deg
    sog[miss], cog[miss] = 102.3, 360.0
    return dict(t_true=t, t_rep=t_rep, pos=pos, sog_kn=sog, cog=cog, is_outlier=is_out)


def write_csv(path, rep, lon0, lat0, rng, dirty=False, mmsi=219000001):
    import pandas as pd
    ll = xy_to_ll(rep['pos'], lon0, lat0)
    df = pd.DataFrame(dict(timestamp=pd.to_datetime(rep['t_rep'], unit='s', utc=True).strftime('%Y-%m-%d %H:%M:%S'),
                           mmsi=mmsi, longitude_degrees=ll[:, 0], latitude_degrees=ll[:, 1],
                           speed=rep['sog_kn'], cog=rep['cog']))
    if dirty:
        dup = df.sample(frac=0.01, random_state=int(rng.integers(1 << 30)))
        df = pd.concat([df, dup]).sort_values('timestamp', kind='stable').reset_index(drop=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def make_fleet(out_dir, n_voyages=24, seed=7, regimes=('raw', 'ds30', 'ds60', 'dirty'), rate_range=(0.15, 0.5),
               dirty_kw=None, **sample_kw):
    """Write the CSV files of every regime and return the truth of each voyage."""
    rng = np.random.default_rng(seed)
    dirty_kw = dict(dirty=True, runs=1, nan_sog=0.05, sentinels=0.01) if dirty_kw is None else dirty_kw
    fleet = []
    for i in range(n_voyages):
        tr = make_truth(rng, rng.uniform(3000.0, 4800.0), rate_range=rate_range)
        tr['lon0'], tr['lat0'] = 11.9 + rng.uniform(-0.05, 0.05), 57.6 + rng.uniform(-0.03, 0.03)
        tr['name'] = 'voyage_%02d' % i
        gaps = [(rng.uniform(0.05, 0.9) * len(tr['t']), rng.uniform(60.0, 600.0)) for _ in range(rng.integers(0, 4))]
        tr['gaps'] = gaps
        tr['reports'] = {}
        for reg in regimes:
            r = np.random.default_rng(seed * 1000 + i * 10 + len(reg))
            kw = dict(sample_kw, **(dirty_kw if reg == 'dirty' else {}))
            rep = sample_reports(r, tr, regime='raw' if reg == 'dirty' else reg, gaps=gaps, **kw)
            tr['reports'][reg] = rep
            write_csv(os.path.join(out_dir, reg, tr['name'] + '.csv'), rep, tr['lon0'], tr['lat0'], r, dirty=(reg == 'dirty'),
                      mmsi=219000001 + i)          # one vessel id per voyage, so the split by vessel is exercised
        fleet.append(tr)
    return fleet


def make_pair(out_dir, seed=11, dcpa=300.0):
    """A crossing pair with a known closest approach: two CSV files in one encounter folder."""
    rng = np.random.default_rng(seed)
    n = 3600
    a = dict(t=np.arange(n, dtype=float), speed=np.full(n, 6.0), course=np.full(n, 90.0), rate=np.zeros(n))
    a['x'], a['y'] = 6.0 * a['t'] - 9000.0, np.zeros(n)
    b = dict(t=np.arange(n, dtype=float), speed=np.full(n, 5.0), course=np.full(n, 0.0), rate=np.zeros(n))
    b['x'], b['y'] = np.zeros(n), 5.0 * (b['t'] - 1500.0) - dcpa * math.hypot(6.0, 5.0) / 6.0
    lon0, lat0 = 11.9, 57.6
    reps = []
    for k, (tr, span) in enumerate(((a, (0, 3000)), (b, (400, 3500)))):
        r = np.random.default_rng(seed + k)
        rep = sample_reports(r, tr, regime='raw' if k == 0 else 'ds30', gaps=[])
        sel = (rep['t_true'] >= span[0]) & (rep['t_true'] <= span[1])
        rep = {key: val[sel] for key, val in rep.items()}
        write_csv(os.path.join(out_dir, 'crossing', '1', '%d.csv' % (k + 1)), rep, lon0, lat0, r, mmsi=219000001 + k)
        reps.append(rep)
    d = np.hypot(a['x'] - b['x'], a['y'] - b['y'])
    return dict(a=a, b=b, lon0=lon0, lat0=lat0, reports=reps, dcpa=float(d.min()), tcpa=float(a['t'][int(np.argmin(d))]))
