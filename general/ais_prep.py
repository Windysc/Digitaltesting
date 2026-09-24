"""
ais_prep.py -- timestamp-correct sampling and smoothing of AIS tracks
(reference implementation of the mending plan for part 1, data extraction).

It replaces the first data step of the repository (data_csv2npy.py: spline over
the row index, noise in degrees; removed on 2026-09-24, its resampling function
is kept in legacy_stage.py) and the point-count smoothing in the environments.
Pure numpy; pandas is used only to read CSV files.

Chain A, generator arrays
    clean -> split the voyages into train / val / test -> outlier rules ->
    split at reception gaps -> Kalman / RTS smoother on the real timestamps
    (strength from the training voyages, inside physical bounds) -> common
    absolute time grid -> windows of fixed duration -> optional augmentation in
    metres (training split only).
Chain B, scenario routes
    window or smoothed track -> C2 spline through the samples -> table at
    uniform arc length with heading and curvature of that same curve ->
    curvature / yaw-rate check at the scenario speed and scale -> playback
    from the table, so what is checked is what is sailed.
Report
    aggregate numbers only (no positions, no absolute times, no latitude), so
    it can be shared while the data stay private.

Command line
    python ais_prep.py prepare --input <csv files or folders> --out <dir>
    python ais_prep.py legacy-check --input <csv files or folders>

Conventions: x east, y north [m]; course in degrees clockwise from north
(AIS COG); route headings in radians, mathematical (0 = east); speeds in m/s.
"""
import argparse
import json
import math
import os

import numpy as np

KN = 0.514444
A_WGS, E2_WGS = 6378137.0, 6.69437999014e-3

DEFAULTS = dict(
    grid_dt=20.0,           # s, the environments' trace_dt
    window_points=100,      # 100 x 20 s = 1980 s between first and last sample
    stride=50,
    gap_s='auto',           # split threshold per voyage: max(gap_floor, gap_factor x its median report interval) <= gap_cap
    gap_floor=60.0,
    gap_factor=2.0,
    gap_cap=120.0,          # s; a 120 s gap that hides a turn costs about 20 m at P95, 180 s about 60 m (test T4)
    sigma_p=10.0,           # m per axis, position measurement SD used by the smoother
    sigma_v=0.2,            # m/s per axis, SOG/COG measurement SD when velocity aiding is on
    q='auto',               # m^2/s^3, spectral density of the acceleration noise
    q_bounds=(1e-3, 1e-2),  # the choice is kept inside this range (see choose_q)
    q_grid=(3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
    q_tolerance=0.02,       # scores within 2 % of the best count as ties; the smoothest setting wins
    cog_source='unknown',   # 'receiver' when SOG / COG are known to be the ship's own GNSS velocity
    max_cog_lag=5.0,        # s, SOG / COG enter the smoother only when the receiver lag is at most this
    v_cap=20.0,             # m/s, absolute plausibility limit for an implied speed
    run_fixes=12,           # a run of bad fixes is at most this long ...
    run_s=180.0,            # ... and lasts at most this many seconds
    gate_floor=40.0,        # m, smallest residual treated as an outlier in the residual pass
    min_speed=0.5,          # m/s, windows whose 5th-percentile speed is lower are dropped
    route_table_ds=5.0,     # m, spacing of the route table
    route_end_fit=300.0,    # m, length of the fit that gives the diagnostic end courses
    yaw_limit=0.5,          # deg/s: 0.5 is the v2 / v3 target default; the encounter standard has 0.28
                            # (maritime) and 1.964 (arena, routes scaled by 0.143)
    route_speed=6.0,        # m/s, scenario speed for the playback check
    split=(0.70, 0.15, 0.15),
    outlier_rules=True,     # False switches both outlier rules off (tests only)
    seed=0,
)

ALIASES = {
    'time': ['timestamp', 'time', 'datetime', 'date_time', 'date_time_utc', 'basedatetime', '# timestamp',
             'time_utc', 'received_time', 'msgtime', 'ts', 't', 'time_s', 'epoch'],
    'lon': ['longitude_degrees', 'longitude', 'lon', 'long', 'lng', 'x'],
    'lat': ['latitude_degrees', 'latitude', 'lat', 'y'],
    'sog': ['speed', 'sog', 'speed_over_ground', 'sog_knots', 'speed_knots', 'sog_kn'],
    'cog': ['cog', 'course', 'course_over_ground', 'course_degrees', 'cog_degrees'],
    'id': ['mmsi', 'vessel_id', 'ship_id', 'imo'],
}
LAGS = (0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0)


# ------------------------------------------------------------------ geodesy
def _ecef(lon_deg, lat_deg):
    lon, lat = np.radians(lon_deg), np.radians(lat_deg)
    n = A_WGS / np.sqrt(1.0 - E2_WGS * np.sin(lat) ** 2)
    return np.stack([n * np.cos(lat) * np.cos(lon), n * np.cos(lat) * np.sin(lon),
                     n * (1.0 - E2_WGS) * np.sin(lat)], axis=-1)


def ll_to_xy(lon, lat, lon0, lat0):
    """WGS84 lon/lat [deg] -> local east/north [m] on the tangent plane at (lon0, lat0)."""
    d = _ecef(np.asarray(lon, float), np.asarray(lat, float)) - _ecef(lon0, lat0)
    l0, p0 = math.radians(lon0), math.radians(lat0)
    e = -math.sin(l0) * d[..., 0] + math.cos(l0) * d[..., 1]
    n = (-math.sin(p0) * math.cos(l0) * d[..., 0] - math.sin(p0) * math.sin(l0) * d[..., 1]
         + math.cos(p0) * d[..., 2])
    return np.stack([e, n], axis=-1)


def xy_to_ll(xy, lon0, lat0):
    """Inverse of ll_to_xy by Newton steps on the local radii (sub-millimetre after 3 steps)."""
    xy = np.asarray(xy, float)
    s = math.sin(math.radians(lat0))
    w = math.sqrt(1.0 - E2_WGS * s * s)
    rn, rm = A_WGS / w, A_WGS * (1.0 - E2_WGS) / w ** 3
    lon = lon0 + np.degrees(xy[..., 0] / (rn * math.cos(math.radians(lat0))))
    lat = lat0 + np.degrees(xy[..., 1] / rm)
    for _ in range(3):
        err = xy - ll_to_xy(lon, lat, lon0, lat0)
        lon = lon + np.degrees(err[..., 0] / (rn * np.cos(np.radians(lat))))
        lat = lat + np.degrees(err[..., 1] / rm)
    return np.stack([lon, lat], axis=-1)


def wrap180(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


# ------------------------------------------------------------------ reading
def _find(cols, kind, override):
    if override and override.get(kind):
        return override[kind]
    low = {c.lower().strip(): c for c in cols}
    for a in ALIASES[kind]:
        if a in low:
            return low[a]
    return None


def _parse_time(col):
    import pandas as pd
    if pd.api.types.is_numeric_dtype(col):
        t = col.to_numpy(float)
        med = np.nanmedian(t)
        return t / 1e6 if med > 1e14 else t / 1e3 if med > 1e11 else t
    dt = pd.to_datetime(col, utc=True, errors='coerce')
    return ((dt - pd.Timestamp('1970-01-01', tz='UTC')) / pd.Timedelta(seconds=1)).to_numpy(float)


def read_tracks_csv(path, columns=None):
    """One CSV -> list of raw tracks (one per vessel id), each a dict of arrays t, lon, lat, sog, cog."""
    import pandas as pd
    df = pd.read_csv(path)
    names = {k: _find(df.columns, k, columns) for k in ALIASES}
    for k in ('lon', 'lat'):
        if names[k] is None:
            raise ValueError('%s: no %s column found (have %s)' % (path, k, list(df.columns)))
    if names['time'] is None:
        raise ValueError('%s: no timestamp column found (have %s). The chain needs the report times; '
                         're-export the track with them.' % (path, list(df.columns)))
    t = _parse_time(df[names['time']])
    out = []
    groups = [(None, df.index)] if names['id'] is None else list(df.groupby(names['id']).groups.items())
    for vid, idx in groups:
        sel = df.loc[idx]
        sog = sel[names['sog']].to_numpy(float) if names['sog'] else None
        cog = sel[names['cog']].to_numpy(float) if names['cog'] else None
        if sog is not None:                                   # AIS "not available" is 102.3 kn; nothing sails faster
            sog = np.where((sog >= 102.2) | (sog < 0), np.nan, sog)
        if cog is not None:                                   # AIS "not available" is 360.0 deg
            cog = np.where((cog >= 360.0) | (cog < 0), np.nan, cog)
        out.append(dict(source=os.path.basename(path), path=os.path.abspath(path),
                        vessel=str(vid) if vid is not None else '',
                        t=t[df.index.get_indexer(idx)], lon=sel[names['lon']].to_numpy(float),
                        lat=sel[names['lat']].to_numpy(float), sog=sog, cog=cog))
    return out


def collect_csv(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, fs in os.walk(p):
                files += [os.path.join(root, f) for f in sorted(fs) if f.lower().endswith('.csv')]
        else:
            files.append(p)
    return files


# ----------------------------------------------------------------- cleaning
def _take(tr, keep):
    out = dict(tr)
    for k in ('t', 'lon', 'lat', 'sog', 'cog', 'xy', 'sog_ms'):
        if out.get(k) is not None:
            out[k] = out[k][keep]
    return out


def clean_track(tr, counts):
    """Sort by time, drop unusable rows and duplicate timestamps."""
    t, lon, lat = tr['t'], tr['lon'], tr['lat']
    ok = np.isfinite(t) & np.isfinite(lon) & np.isfinite(lat) & (np.abs(lat) <= 90) & (np.abs(lon) <= 180)
    ok &= ~((np.abs(lat) < 1e-9) & (np.abs(lon) < 1e-9))
    counts['raw_reports'] += len(t)
    counts['unusable_rows'] += int(len(t) - ok.sum())
    counts['non_monotonic_rows'] += int(np.sum(np.diff(t[ok]) < 0))
    idx = np.flatnonzero(ok)[np.argsort(t[ok], kind='stable')]
    tr = _take(tr, idx)
    keep = np.concatenate([[True], np.diff(tr['t']) > 0])
    counts['duplicate_timestamps'] += int(np.sum(~keep))
    return _take(tr, keep)


def drop_stale(tr, counts):
    """A position repeated exactly although the reported speed says the ship moved more than 10 m since the
    last report is a stale report.  Slow ships and coarse coordinate rounding repeat positions legitimately,
    so nothing is dropped without a speed."""
    if tr.get('sog_ms') is None or len(tr['t']) < 3:
        return tr
    same = np.concatenate([[False], (np.diff(tr['lon']) == 0) & (np.diff(tr['lat']) == 0)])
    moved = np.concatenate([[0.0], np.diff(tr['t'])]) * np.nan_to_num(tr['sog_ms'], nan=0.0)
    stale = same & (moved > 10.0)
    counts['stale_repeats'] += int(stale.sum())
    return _take(tr, ~stale)


def detect_sog_unit(t, xy, sog):
    """Knots or m/s, from the ratio of position-implied speed to SOG over baselines of one to two minutes."""
    if sog is None or len(t) < 10:
        return None, float('nan')
    ratios = []
    j = 0
    for i in range(len(t)):
        while j < len(t) - 1 and t[j] - t[i] < 60.0:
            j += 1
        if 60.0 <= t[j] - t[i] < 120.0 and np.isfinite(sog[i:j + 1]).any():
            s = np.nanmean(sog[i:j + 1])
            if s > 1.0:
                ratios.append(np.linalg.norm(xy[j] - xy[i]) / (t[j] - t[i]) / s)
    if len(ratios) < 5:
        return None, float('nan')
    r = float(np.median(ratios))
    return ('kn' if 0.40 < r < 0.65 else 'ms' if 0.85 < r < 1.20 else None), r


def distance_outliers(t, xy, sog=None, v_cap=20.0, sigma_p=10.0, k=4.0, run_fixes=12, run_s=180.0):
    """Outlier rule in distance form, for single bad fixes and for runs of them.

    A step between two fixes is implausible when its length exceeds v * dt + k * sqrt(2) * sigma_p, with v
    the plausible speed (1.5 x SOG + 2 m/s when SOG is there, never above v_cap).  The noise term keeps short
    report intervals from raising false alarms.  Bad fixes sit between two implausible steps (into the
    offset and back) whose outer fixes are plausible with each other; everything between them is removed,
    up to run_fixes fixes or run_s seconds.  A first or last fix with one implausible step is removed.
    An implausible step that never closes is a lasting jump (another receiver, another ship under the same
    id); the track is cut there.  Returns (bad mask, cut mask: a cut follows fix i).
    """
    n = len(t)
    bad, cut = np.zeros(n, bool), np.zeros(n, bool)
    slack = k * math.sqrt(2.0) * sigma_p
    sg = None if sog is None else np.where(np.isfinite(sog), sog, v_cap)

    def plausible(a, b):
        v = v_cap if sg is None else min(v_cap, 1.5 * max(sg[a], sg[b]) + 2.0)
        return np.linalg.norm(xy[b] - xy[a]) <= v * (t[b] - t[a]) + slack

    for _ in range(6):
        idx = np.flatnonzero(~bad)
        if len(idx) < 4:
            break
        imp = np.array([not plausible(idx[i], idx[i + 1]) for i in range(len(idx) - 1)])
        steps = np.flatnonzero(imp)
        new = np.zeros(len(idx), bool)
        open_steps = []
        a = 0
        while a < len(steps):
            s0, closed = steps[a], False
            for b in range(a + 1, len(steps)):
                s1 = steps[b]
                if s1 - s0 > run_fixes or t[idx[s1 + 1]] - t[idx[s0]] > run_s:
                    break
                if plausible(idx[s0], idx[s1 + 1]):
                    new[s0 + 1:s1 + 1] = True
                    a, closed = b + 1, True
                    break
            if not closed:
                if s0 == 0 and not imp[1]:
                    new[0] = True
                elif s0 == len(idx) - 2 and not imp[-2]:
                    new[-1] = True
                else:
                    open_steps.append(s0)
                a += 1
        if not new.any():
            cut[idx[open_steps]] = True
            break
        bad[idx[new]] = True
    return bad, cut


# ----------------------------------------------------------------- smoother
def rts_smooth(t, z, t_eval, q, sigma_p, vel=None, sigma_v=0.2, use=None):
    """Kalman filter + Rauch-Tung-Striebel smoother, continuous white-noise-acceleration model.

    t (n,) strictly increasing report times, z (n, 2) positions [m], vel (n, 2) optional velocity
    measurements [m/s] (rows with nan are position-only), use (n,) bool mask of the fixes that enter the
    filter.  q [m^2/s^3] is the spectral density of the acceleration noise: larger q follows manoeuvres
    more closely and passes more noise.  The two axes share one covariance, so the covariance recursion
    is scalar.  Returns positions, velocities and the position SD at t_eval and at every fix time.  The SD
    is the model's own figure (white errors of size sigma_p), not a calibrated uncertainty.
    """
    t = np.asarray(t, float)
    z = np.asarray(z, float)
    n = len(t)
    use = np.ones(n, bool) if use is None else np.asarray(use, bool)
    t_eval = np.asarray(t_eval, float)
    T = np.union1d(t, t_eval)
    N = len(T)
    at_fix = np.searchsorted(T, t)
    at_eval = np.searchsorted(T, t_eval)
    meas = -np.ones(N, int)
    meas[at_fix[use]] = np.flatnonzero(use)
    rp, rv = sigma_p ** 2, sigma_v ** 2
    first = int(np.flatnonzero(use)[0])
    xp = np.zeros((N, 2)); xv = np.zeros((N, 2))                # filtered
    pp = np.zeros((N, 2)); pv = np.zeros((N, 2))                # predicted
    A = np.zeros(N); B = np.zeros(N); C = np.zeros(N)           # filtered covariance [[A, B], [B, C]]
    Ap = np.zeros(N); Bp = np.zeros(N); Cp = np.zeros(N)        # predicted covariance
    p = z[first].copy()
    v = vel[first].copy() if vel is not None and np.all(np.isfinite(vel[first])) else np.zeros(2)
    a, b, c = 1e6, 0.0, (1e2 if vel is None else 25.0)
    for k in range(N):
        if k > 0:
            dt = T[k] - T[k - 1]
            p = p + v * dt
            a, b, c = (a + 2.0 * b * dt + c * dt * dt + q * dt ** 3 / 3.0,
                       b + c * dt + q * dt * dt / 2.0, c + q * dt)
        pp[k], pv[k], Ap[k], Bp[k], Cp[k] = p, v, a, b, c
        i = meas[k]
        if i >= 0:
            nu = z[i] - p
            if vel is not None and np.all(np.isfinite(vel[i])):
                nv = vel[i] - v
                det = (a + rp) * (c + rv) - b * b
                k00, k01 = (a * (c + rv) - b * b) / det, b * rp / det
                k10, k11 = b * rv / det, (c * (a + rp) - b * b) / det
                p = p + k00 * nu + k01 * nv
                v = v + k10 * nu + k11 * nv
                a, b, c = (1 - k00) * a - k01 * b, (1 - k00) * b - k01 * c, -k10 * b + (1 - k11) * c
            else:
                s = a + rp
                k0, k1 = a / s, b / s
                p = p + k0 * nu
                v = v + k1 * nu
                a, b, c = a * (1 - k0), b * (1 - k0), c - b * b / s
        xp[k], xv[k], A[k], B[k], C[k] = p, v, a, b, c
    sp, sv = xp.copy(), xv.copy()
    SA, SB, SC = A.copy(), B.copy(), C.copy()
    for k in range(N - 2, -1, -1):
        dt = T[k + 1] - T[k]
        m00, m01, m10, m11 = A[k] + B[k] * dt, B[k], B[k] + C[k] * dt, C[k]
        det = Ap[k + 1] * Cp[k + 1] - Bp[k + 1] ** 2
        c00 = (m00 * Cp[k + 1] - m01 * Bp[k + 1]) / det
        c01 = (-m00 * Bp[k + 1] + m01 * Ap[k + 1]) / det
        c10 = (m10 * Cp[k + 1] - m11 * Bp[k + 1]) / det
        c11 = (-m10 * Bp[k + 1] + m11 * Ap[k + 1]) / det
        dp, dv = sp[k + 1] - pp[k + 1], sv[k + 1] - pv[k + 1]
        sp[k] = xp[k] + c00 * dp + c01 * dv
        sv[k] = xv[k] + c10 * dp + c11 * dv
        da, db, dc = SA[k + 1] - Ap[k + 1], SB[k + 1] - Bp[k + 1], SC[k + 1] - Cp[k + 1]
        SA[k] = A[k] + c00 * c00 * da + 2 * c00 * c01 * db + c01 * c01 * dc
        SB[k] = B[k] + c00 * c10 * da + (c00 * c11 + c01 * c10) * db + c01 * c11 * dc
        SC[k] = C[k] + c10 * c10 * da + 2 * c10 * c11 * db + c11 * c11 * dc
    return dict(pos=sp[at_eval], vel=sv[at_eval], sd=np.sqrt(np.maximum(SA[at_eval], 0.0)),
                pos_fix=sp[at_fix], vel_fix=sv[at_fix], innov=(z - pp[at_fix]))


def residual_outliers(t, xy, use, q, sigma_p, vel, sigma_v, gate_floor):
    """Residual pass for what the distance rule lets through.

    Bad fixes drag the smoothed track, so their good neighbours also show large residuals in the first
    pass.  They are removed with the bad ones and taken back once the track no longer bends towards them.
    """
    start = use.copy()
    use = use.copy()
    for _ in range(4):
        if use.sum() < 5:
            break
        sm = rts_smooth(t, xy, t[:1], q, sigma_p, vel, sigma_v, use)
        res = np.linalg.norm(xy - sm['pos_fix'], axis=1)
        med = np.median(res[use])
        thr = max(gate_floor, med + 6.0 * 1.4826 * np.median(np.abs(res[use] - med)))
        back = start & ~use & (res <= thr)
        new = use & (res > thr)
        if not new.any() and not back.any():
            break
        use = (use & ~new) | back
    return use


def track_frame(res, vel, vmin=0.5):
    """Residual vectors -> (along, cross) in the frame of the smoothed velocity; slow fixes are dropped."""
    sp = np.linalg.norm(vel, axis=1)
    ok = sp > vmin
    u = vel[ok] / sp[ok, None]
    return np.sum(res[ok] * u, axis=1), res[ok, 1] * u[:, 0] - res[ok, 0] * u[:, 1], ok


def course_deg(vel):
    return np.degrees(np.arctan2(vel[..., 0], vel[..., 1])) % 360.0


def velocity_from_sogcog(sog_ms, cog_deg):
    c = np.radians(cog_deg)
    return np.stack([sog_ms * np.sin(c), sog_ms * np.cos(c)], axis=-1)


def cog_turning(t, cog, half_s=30.0, rate=0.1):
    """Turning (1) / straight (0) from the reported COG alone (change over about a minute), so the label does
    not come from the track that is being judged.  nan where COG is missing."""
    out = np.full(len(t), np.nan)
    i = np.arange(len(t))
    lo = np.minimum(np.searchsorted(t, t - half_s), np.maximum(i - 1, 0))          # at least the neighbours,
    hi = np.maximum(np.minimum(np.searchsorted(t, t + half_s, side='right') - 1, len(t) - 1),
                    np.minimum(i + 1, len(t) - 1))                                  # so sparse reports still get a label
    ok = (hi > lo) & np.isfinite(cog[lo]) & np.isfinite(cog[hi])
    out[ok] = (np.abs(wrap180(cog[hi] - cog[lo])) / np.maximum(t[hi] - t[lo], 1.0) > rate)[ok]
    return out


# ------------------------------------------------------------ holdout tuning
def holdout_error(t, xy, use, q, sigma_p, vel, sigma_v, rng, block_s=(20.0, 60.0), share=0.10):
    """Hide single fixes and short blocks, smooth the rest, return the prediction errors [m]."""
    n = len(t)
    cand = np.flatnonzero(use)[2:-2]
    if len(cand) < 20:
        return np.zeros(0), np.zeros(0)
    hide_s = rng.choice(cand, size=max(1, int(share * len(cand))), replace=False)
    m1 = use.copy(); m1[hide_s] = False
    e_single = np.linalg.norm(rts_smooth(t, xy, t[hide_s], q, sigma_p, vel, sigma_v, m1)['pos'] - xy[hide_s], axis=1)
    hidden = np.zeros(n, bool)
    span, tries = t[cand[-1]] - t[cand[0]], 0
    while hidden.sum() < share * len(cand) and tries < 200:
        tries += 1
        t0 = t[cand[0]] + rng.uniform(0.0, span)
        blk = use & (t >= t0) & (t <= t0 + rng.uniform(*block_s))
        blk[:cand[0]] = False; blk[cand[-1] + 1:] = False
        hidden |= blk
    m2 = use & ~hidden
    if hidden.sum() == 0 or m2.sum() < 5:
        return e_single, np.zeros(0)
    idx = np.flatnonzero(hidden)
    e_block = np.linalg.norm(rts_smooth(t, xy, t[idx], q, sigma_p, vel, sigma_v, m2)['pos'] - xy[idx], axis=1)
    return e_single, e_block


def choose_q(segments, cfg, rng):
    """Smoothing strength from the data, inside physical bounds.

    What the tests showed (T3, check_cog_rule.py):
    * The error against the truth is flat over about a decade of q, and every truth criterion (position,
      course, turn rate) prefers the same value: 0.001 to 0.01 m^2/s^3 across report rates and noise types.
      That range is also what ship motion implies: a turn of 0.3 deg/s at 6 m/s is a lateral acceleration
      of 0.03 m/s^2.
    * A blocked holdout is a weak referee.  Position errors are correlated in time, so a hidden fix is
      predicted a little better by a track that follows the error of its neighbours; the plain minimum
      under-smooths by up to a factor of 10 and much more with sparse reports.  Long hidden blocks with
      guard bands do worse: their score is set by the few blocks that hide a turn.
    * A COG from the ship's own receiver is an independent referee, because it comes from the Doppler
      velocity and does not share the position error.  It cannot be told from the data alone whether a
      COG column is that (a provider may have computed it from smoothed positions), so it counts only
      when cfg['cog_source'] == 'receiver', and then only if the two tests of cog_diagnostics pass and
      the hidden fixes are predicted at least as well with SOG / COG in the smoother as without.
    The rule: scores within q_tolerance of the best are ties and the smallest such q is taken; both
    referees can only err towards a large q, so the smaller of their choices is taken; the result is kept
    inside q_bounds.  The table lists the whole grid so the report shows when a bound is active.
    """
    table, per_voyage = [], {}
    seeds = rng.integers(1 << 30, size=len(segments))                  # the same hidden fixes for every q
    for q in cfg['q_grid']:
        es, eb, dc, dv = [], [], [], []
        for sg, sd in zip(segments, seeds):
            a, b = holdout_error(sg['t'], sg['xy'], sg['use'], q, cfg['sigma_p'], None, cfg['sigma_v'],
                                 np.random.default_rng(sd))
            es.append(a); eb.append(b)
            per_voyage.setdefault(sg['voyage'], {}).setdefault(q, []).append(np.concatenate([a, b]))
            if sg['sog'] is not None and sg['cog'] is not None:
                sm = rts_smooth(sg['t'], sg['xy'], sg['t'][:1], q, cfg['sigma_p'], None, cfg['sigma_v'], sg['use'])
                good = sg['use'] & np.isfinite(sg['cog']) & (np.nan_to_num(sg['sog']) > 1.5)
                dc.append(wrap180(course_deg(sm['vel_fix']) - sg['cog'])[good])
                dv.append((np.linalg.norm(sm['vel_fix'], axis=1) - sg['sog'])[good])
        es, eb = _cat(es), _cat(eb)
        row = dict(q=q, single_rms=_rms(es), single_p95=_pct(es, 95), block_rms=_rms(eb),
                   block_p95=_pct(eb, 95), n_single=len(es), n_block=len(eb))
        if len(_cat(dc)):
            row.update(course_vs_cog_rms_deg=_rms(_cat(dc)), speed_vs_sog_rms_ms=_rms(_cat(dv)))
        table.append(row)
    tol = 1.0 + cfg['q_tolerance']
    score = np.array([r['single_rms'] + r['block_rms'] for r in table])
    lo, hi = cfg['q_bounds']
    clip = lambda q: float(min(max(q, lo), hi))
    info = dict(rule='too little data: geometric mean of the bounds', q_holdout=None, q_cog=None, cog_independent=None,
                cog_lag_s=None, cog_usable=False, cog_source=cfg['cog_source'], bounds=[lo, hi], bound_active=False)
    if not np.isfinite(score).any():
        return float(math.sqrt(lo * hi)), table, info
    info['q_holdout'] = table[int(np.flatnonzero(score <= np.nanmin(score) * tol)[0])]['q']
    best = []
    for d in per_voyage.values():
        s = np.array([_rms(_cat(d[q])) for q in cfg['q_grid']])
        if np.isfinite(s).any():
            best.append(cfg['q_grid'][int(np.nanargmin(s))])
    info['per_voyage_holdout_minimum'] = {('%g' % q): int(np.sum(np.asarray(best) == q)) for q in cfg['q_grid']}
    q, info['rule'] = info['q_holdout'], 'holdout, ties to the smaller q'
    c = np.array([r.get('course_vs_cog_rms_deg', np.nan) for r in table])
    if np.isfinite(c).any():
        info['q_cog'] = table[int(np.flatnonzero(c <= np.nanmin(c) * tol)[0])]['q']
        info.update(cog_diagnostics(segments, info['q_cog'], cfg))
        lag_ok = info['cog_lag_s'] is not None and info['cog_lag_s'] <= cfg['max_cog_lag']
        info['cog_usable'] = bool(cfg['cog_source'] == 'receiver' and info['cog_independent'] and lag_ok)
        if info['cog_usable']:
            # last test: with SOG / COG in the smoother the hidden fixes must be predicted at least as well.
            # A course taken from smoothed positions passes the two tests above and fails this one.
            qa = clip(min(q, info['q_cog']))
            ea, ep = [], []
            for sg, sd in zip(segments, seeds):
                if sg['vel_meas'] is None:
                    continue
                for vel, acc in ((sg['vel_meas'], ea), (None, ep)):
                    a, b = holdout_error(sg['t'], sg['xy'], sg['use'], qa, cfg['sigma_p'], vel, cfg['sigma_v'],
                                         np.random.default_rng(sd))
                    acc.append(np.concatenate([a, b]))
            info['holdout_rms_with_sogcog_over_positions_only'] = _rms(_cat(ea)) / max(_rms(_cat(ep)), 1e-9)
            info['cog_usable'] = bool(info['holdout_rms_with_sogcog_over_positions_only'] <= 1.0)
        if info['cog_usable']:
            q, info['rule'] = min(q, info['q_cog']), 'smaller of the holdout and the COG choice'
    info['bound_active'] = bool(clip(q) != q)
    return clip(q), table, info


def cog_diagnostics(segments, q, cfg):
    """Two tests of the reported COG against a positions-only smooth at strength q.

    Independence.  A COG that was computed from consecutive fixes repeats their error: on straight
    stretches its deviation from the smooth moves together with the deviation of the raw fix-to-fix
    course (correlation 1 for one step, 0.5 for a course over several fixes, which still shares one fix
    with the raw course).  The test cannot see a COG computed from positions that were smoothed first, so
    a pass is no proof of a receiver COG; that has to be known from the data source.
    Lag.  Receivers low-pass their velocity output.  The smoother is symmetric in time and has no lag of
    its own, so the shift that brings the smoothed course closest to COG estimates the receiver lag.
    """
    d_cog, d_raw, by_lag, strata = [], [], {lag: [] for lag in LAGS}, {}
    for sg in segments:
        if sg['sog'] is None or sg['cog'] is None or sg['use'].sum() < 12:
            continue
        t, u = sg['t'], np.flatnonzero(sg['use'])
        sog = np.nan_to_num(sg['sog'])
        sm = rts_smooth(t, sg['xy'], t[:1], q, cfg['sigma_p'], None, cfg['sigma_v'], sg['use'])
        crs = course_deg(sm['vel_fix'])
        turning = cog_turning(t, sg['cog'])
        step = sg['xy'][u[1:]] - sg['xy'][u[:-1]]
        k = u[1:]
        ok = np.isfinite(sg['cog'][k]) & (sog[k] > 1.5) & (np.linalg.norm(step, axis=1) > 20.0)
        straight = ok & (turning[k] == 0)
        d_cog.append(wrap180(sg['cog'][k] - crs[k])[straight])
        d_raw.append(wrap180(np.degrees(np.arctan2(step[:, 0], step[:, 1])) - crs[k])[straight])
        good = sg['use'] & np.isfinite(sg['cog']) & (sog > 1.5)
        dev = wrap180(crs - sg['cog'])
        for name, band in (('below 3 m/s', (0.0, 3.0)), ('3 to 6 m/s', (3.0, 6.0)), ('above 6 m/s', (6.0, 99.0))):
            for label, val in (('straight', 0), ('turning', 1)):
                m = good & (sog >= band[0]) & (sog < band[1]) & (turning == val)
                strata.setdefault('%s, %s' % (name, label), []).append(dev[m])
        if good.any():
            for lag in LAGS:
                te = np.clip(t[good] - lag, t[u[0]], t[u[-1]])
                o = np.argsort(te, kind='stable')
                vel = rts_smooth(t, sg['xy'], te[o], q, cfg['sigma_p'], None, cfg['sigma_v'], sg['use'])['vel']
                by_lag[lag].append(wrap180(course_deg(vel) - sg['cog'][good][o]))
    d_cog, d_raw = _cat(d_cog), _cat(d_raw)
    out = dict(cog_independent=None, cog_lag_s=None)
    if len(d_cog) > 30:
        diff = _rms(wrap180(d_cog - d_raw))
        corr = float(np.corrcoef(d_cog, d_raw)[0, 1]) if d_cog.std() > 0 and d_raw.std() > 0 else 1.0
        out.update(cog_vs_raw_position_course_rms_deg=diff, cog_raw_correlation_on_straights=corr,
                   cog_independent=bool(diff > 0.5 and corr < 0.3))
    lag_rms = {lag: _rms(_cat(v)) for lag, v in by_lag.items() if len(_cat(v))}
    if lag_rms:
        best = min(lag_rms.values())
        out['cog_lag_s'] = float(min(lag for lag, r in lag_rms.items() if r <= best * 1.01))
        out['course_vs_cog_rms_by_lag_deg'] = {('%g' % lag): round(r, 3) for lag, r in lag_rms.items()}
    out['course_vs_cog_by_speed'] = {k: _stats(_cat(v)) for k, v in strata.items() if len(_cat(v)) >= 20}
    return out


def _rms(a):
    return float(np.sqrt(np.mean(np.square(a)))) if len(a) else float('nan')


def _pct(a, p):
    return float(np.percentile(a, p)) if len(a) else float('nan')


def _cat(lst):
    lst = [np.asarray(a) for a in lst if len(a)]
    return np.concatenate(lst) if lst else np.zeros(0)


# ------------------------------------------------------------------- routes
def _spline_second_derivatives(u, y):
    """Second derivatives of the cubic spline through (u, y) with not-a-knot ends (y may have columns)."""
    n = len(u)
    h = np.diff(u)
    Amat = np.zeros((n, n))
    rhs = np.zeros((n,) + y.shape[1:])
    for i in range(1, n - 1):
        Amat[i, i - 1], Amat[i, i], Amat[i, i + 1] = h[i - 1], 2.0 * (h[i - 1] + h[i]), h[i]
        rhs[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    Amat[0, 0], Amat[0, 1], Amat[0, 2] = h[1], -(h[0] + h[1]), h[0]
    Amat[-1, -3], Amat[-1, -2], Amat[-1, -1] = h[-1], -(h[-2] + h[-1]), h[-2]
    return np.linalg.solve(Amat, rhs)


def build_route(xy, table_ds=5.0, end_fit=300.0, smooth_m=0.0):
    """Track samples (a 20 s window or a dense smoothed track) -> route table at uniform arc length.

    A C2 cubic spline (not-a-knot ends) runs through the samples in chord length, so neither the corners of
    the sample polygon nor curvature jumps reach the route.  Position, heading and curvature of that one
    curve are tabulated every `table_ds` metres, and the playback (route_eval) reads the same table: the
    curvature that is checked is the curvature that is sailed.  With smooth_m > 0 the tabulated positions
    are smoothed over arc length (Gaussian, end points kept) and heading / curvature are taken from the
    smoothed table.  course0 / course1 are least-squares courses over the first / last `end_fit` metres,
    kept as a diagnostic of how much one noisy end segment could matter.
    Returns dict(s, xy, psi, kappa, ds, course0, course1); angles in radians, mathematical convention.
    """
    xy = np.asarray(xy, float)
    keep = np.concatenate([[True], np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1.0])
    xy = xy[keep]
    u = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    if len(xy) < 5 or u[-1] < 20 * table_ds:
        raise ValueError('track too short for a route: %d points, %.0f m' % (len(xy), u[-1]))
    M = _spline_second_derivatives(u, xy)
    uu = np.linspace(0.0, u[-1], int(math.ceil(u[-1] / (table_ds / 2.0))) + 1)
    i = np.clip(np.searchsorted(u, uu, side='right') - 1, 0, len(u) - 2)
    h = (u[i + 1] - u[i])[:, None]
    a, b = (u[i + 1] - uu)[:, None] / h, (uu - u[i])[:, None] / h
    P = a * xy[i] + b * xy[i + 1] + ((a ** 3 - a) * M[i] + (b ** 3 - b) * M[i + 1]) * h * h / 6.0
    D1 = (xy[i + 1] - xy[i]) / h - (3 * a * a - 1) / 6.0 * h * M[i] + (3 * b * b - 1) / 6.0 * h * M[i + 1]
    D2 = a * M[i] + b * M[i + 1]
    sp = np.maximum(np.linalg.norm(D1, axis=1), 1e-12)
    kap = (D1[:, 0] * D2[:, 1] - D1[:, 1] * D2[:, 0]) / sp ** 3
    psi = np.unwrap(np.arctan2(D1[:, 1], D1[:, 0]))
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (sp[1:] + sp[:-1]) * np.diff(uu))])
    n = max(int(round(cum[-1] / table_ds)), 10)
    ds = cum[-1] / n
    s = ds * np.arange(n + 1)
    pts = np.stack([np.interp(s, cum, P[:, 0]), np.interp(s, cum, P[:, 1])], axis=1)
    psi, kap = np.interp(s, cum, psi), np.interp(s, cum, kap)
    if smooth_m > 0:
        hw = min(int(math.ceil(3 * smooth_m / ds)), len(pts) - 2)
        k = np.exp(-0.5 * (np.arange(-hw, hw + 1) * ds / smooth_m) ** 2)
        k /= k.sum()
        pad = np.concatenate([2 * pts[0] - pts[hw:0:-1], pts, 2 * pts[-1] - pts[-2:-hw - 2:-1]])
        pts = np.stack([np.convolve(pad[:, j], k, mode='valid') for j in range(2)], axis=1)
        cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        n = max(int(round(cum[-1] / table_ds)), 10)
        ds = cum[-1] / n
        s = ds * np.arange(n + 1)
        pts = np.stack([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])], axis=1)
        d = np.gradient(pts, ds, axis=0)
        psi = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        kap = np.gradient(psi, ds)
    m = max(4, int(round(end_fit / ds)) + 1)
    return dict(s=s, xy=pts, psi=psi, kappa=kap, ds=float(ds),
                course0=_fit_course(pts[:m], ds, 0), course1=_fit_course(pts[-m:], ds, -1))


def _fit_course(p, ds, at, bend_deg=3.0):
    """Course at the first (at=0) or last (at=-1) of uniformly spaced points from a least-squares fit in
    arc length: a straight line, or a parabola when the points bend by more than bend_deg (a line lags by
    about curvature x half its length in a turn, a parabola has no lag and about 1.4 times the noise)."""
    k = np.arange(len(p), dtype=float) * ds
    k = k - k[at]
    fit = lambda o: np.linalg.lstsq(np.vander(k, o + 1, increasing=True), p, rcond=None)[0]
    c = fit(2)
    bend = abs(c[1, 0] * 2 * c[2, 1] - c[1, 1] * 2 * c[2, 0]) / max(np.dot(c[1], c[1]), 1e-12) * (k.max() - k.min())
    d = c[1] if math.degrees(bend) > bend_deg else fit(1)[1]
    return math.atan2(d[1], d[0])


def route_eval(route, s):
    """Position, heading [rad] and curvature [1/m] at arc length s (clamped), read from the route table."""
    s = np.clip(np.asarray(s, float), 0.0, route['s'][-1])
    p = np.stack([np.interp(s, route['s'], route['xy'][:, 0]), np.interp(s, route['s'], route['xy'][:, 1])], axis=-1)
    return p, np.interp(s, route['s'], route['psi']), np.interp(s, route['s'], route['kappa'])


def place_route(route, start, course_rad, scale=1.0):
    """Scale a route by the scenario's length scale, turn it so that its own start tangent points along
    course_rad, and move it to start.  The start tangent is the one the playback uses, so the first
    heading of the target equals the scenario course exactly.  Curvature grows by 1 / scale."""
    a = course_rad - route['psi'][0]
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    out = dict(route)
    out['xy'] = (route['xy'] - route['xy'][0]) * scale @ R.T + np.asarray(start, float)
    out['s'], out['ds'] = route['s'] * scale, route['ds'] * scale
    out['psi'], out['kappa'] = route['psi'] + a, route['kappa'] / scale
    out['course0'], out['course1'] = route['course0'] + a, route['course1'] + a
    return out


def extend_route(route, length):
    """Continue the route straight along its own end tangent (C1) by `length` metres."""
    n = int(math.ceil(length / route['ds']))
    d = np.array([math.cos(route['psi'][-1]), math.sin(route['psi'][-1])])
    step = route['ds'] * np.arange(1, n + 1)
    out = dict(route)
    out['xy'] = np.concatenate([route['xy'], route['xy'][-1] + d[None, :] * step[:, None]])
    out['s'] = np.concatenate([route['s'], route['s'][-1] + step])
    out['psi'] = np.concatenate([route['psi'], np.full(n, route['psi'][-1])])
    out['kappa'] = np.concatenate([route['kappa'], np.zeros(n)])
    return out


def fit_route_to_limit(xy, speed, yaw_limit_deg, scale=1.0, table_ds=5.0, end_fit=300.0, max_dev=50.0,
                       ladder=(0.0, 50.0, 100.0, 150.0, 200.0, 300.0)):
    """Smallest spatial smoothing (metres at full size) that keeps speed x curvature of the scaled route
    under the yaw-rate limit.  Returns (route at full size, info).  info['ok'] is False when the limit
    cannot be met without moving the route by more than max_dev metres (the larger of the two one-sided
    distances between the smoothed and the unsmoothed route, i.e. their Hausdorff distance on the tables);
    the route should then not be used at this speed and scale."""
    base = build_route(xy, table_ds, end_fit, 0.0)
    r, yaw, dev, sm = base, 0.0, 0.0, 0.0
    for sm in ladder:
        r = base if sm == 0 else build_route(xy, table_ds, end_fit, sm)
        yaw = math.degrees(speed * float(np.max(np.abs(r['kappa']))) / scale)
        dev = float(max(np.max(perpendicular_distance(r['xy'][::4], base['xy'])),
                        np.max(perpendicular_distance(base['xy'][::4], r['xy'])))) if sm else 0.0
        if yaw <= yaw_limit_deg:
            return r, dict(ok=dev <= max_dev, smooth_m=sm, max_yaw_deg_s=yaw, moved_m=dev)
    return r, dict(ok=False, smooth_m=sm, max_yaw_deg_s=yaw, moved_m=dev)


def perpendicular_distance(points, poly):
    """Distance from each point to a polyline [m]."""
    a, b = poly[:-1], poly[1:]
    ab = b - a
    l2 = np.maximum(np.sum(ab * ab, axis=1), 1e-12)
    out = np.empty(len(points))
    for j, p in enumerate(points):
        f = np.clip(np.sum((p - a) * ab, axis=1) / l2, 0.0, 1.0)
        out[j] = np.min(np.linalg.norm(a + ab * f[:, None] - p, axis=1))
    return out


def chord_deviation(xy):
    """Largest perpendicular distance of a track from its chord, over the whole track [m]."""
    ch = xy[-1] - xy[0]
    n = np.array([-ch[1], ch[0]]) / max(np.linalg.norm(ch), 1e-9)
    return float(np.max(np.abs((xy - xy[0]) @ n)))


# ------------------------------------------------------------- augmentation
def window_kinematics(xy, dt):
    """Speed, acceleration and turn rate series of a window from finite differences on its grid."""
    d = np.diff(xy, axis=0)
    sp = np.linalg.norm(d, axis=1) / dt
    hd = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    return sp, np.diff(sp) / dt, np.degrees(np.diff(hd)) / dt


def augment_window(seg, i0, cfg, rng, amp=(10.0, 20.0), corr_s=(120.0, 300.0), speed=(0.95, 1.05)):
    """One variant of the window that starts at grid index i0 of a segment: the same path sailed a few
    per cent faster or slower (time warp through the smoother) plus a smooth lateral offset.  `amp` is
    the standard deviation of that offset; its largest value inside a window is two to three times that."""
    W, dt = cfg['window_points'], cfg['grid_dt']
    t0 = seg['tg'][i0]
    room = (seg['t'][seg['use']][-1] - t0) / ((W - 1) * dt)
    f = min(rng.uniform(*speed), room)
    tw = t0 + f * dt * np.arange(W)
    sm = rts_smooth(seg['t'], seg['xy'], tw, cfg['q_used'], cfg['sigma_p'], seg['vel'], cfg['sigma_v'], seg['use'])
    pos, vel = sm['pos'], sm['vel'] * f
    tt = dt * np.arange(W)
    ell, a = rng.uniform(*corr_s), rng.uniform(*amp)
    K = a * a * np.exp(-0.5 * ((tt[:, None] - tt[None, :]) / ell) ** 2) + 1e-8 * a * a * np.eye(W)
    off = np.linalg.cholesky(K) @ rng.standard_normal(W)
    sp = np.maximum(np.linalg.norm(vel, axis=1, keepdims=True), 1e-6)
    normal = np.stack([-vel[:, 1], vel[:, 0]], axis=1) / sp
    return pos + normal * off[:, None]


# ------------------------------------------------------------------ pipeline
def _make_segments(vi, v, keep, cut, gap_s, counts):
    """Drop the removed fixes, split at report intervals above gap_s and at cuts, return usable segments."""
    idx = np.flatnonzero(keep)
    brk = np.diff(v['t'][idx]) > gap_s
    if cut is not None and cut.any():
        c = np.concatenate([[0], np.cumsum(cut.astype(int))])       # c[i] = cuts strictly before fix i
        brk |= c[idx[1:]] != c[idx[:-1]]
    edges = np.concatenate([[0], np.flatnonzero(brk) + 1, [len(idx)]])
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 8:
            counts['segments_too_short'] += 1
            continue
        k = idx[a:b]
        out.append(dict(voyage=vi, t=v['t'][k], xy=v['xy'][k], use=np.ones(len(k), bool), vel=None,
                        sog=None if v.get('sog_ms') is None else v['sog_ms'][k],
                        cog=None if v.get('cog') is None else v['cog'][k],
                        vel_meas=None if v.get('vel_meas') is None else v['vel_meas'][k]))
    return out


def prepare(paths, out_dir, columns=None, pairs=False, use_sogcog='auto', augment=0, keep=False, **overrides):
    """Run chain A on CSV files or folders, write the arrays and the aggregate report to out_dir.
    keep=True also returns the internal voyages / segments (for tests against a known truth)."""
    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    rng = np.random.default_rng(cfg['seed'])
    counts = dict(files=0, voyages=0, raw_reports=0, unusable_rows=0, non_monotonic_rows=0,
                  duplicate_timestamps=0, stale_repeats=0, sog_or_cog_not_available=0, removed_by_distance_rule=0,
                  removed_by_residual_pass=0, cuts_at_lasting_jumps=0, segments=0, segments_too_short=0,
                  windows=0, windows_too_slow=0)
    voyages = []
    for f in collect_csv(paths):
        counts['files'] += 1
        for tr in read_tracks_csv(f, columns):
            tr = clean_track(tr, counts)
            if len(tr['t']) >= 10:
                tr['group'] = (os.path.dirname(tr['path']) if pairs else
                               'vessel ' + tr['vessel'] if tr['vessel'] else tr['path'])
                voyages.append(tr)
    counts['voyages'] = len(voyages)
    if not voyages:
        raise ValueError('no usable tracks in %s' % paths)
    group_key = 'folder' if pairs else 'vessel id' if any(v['vessel'] for v in voyages) else 'file'

    # split by group BEFORE anything is estimated from the data, including the SOG-unit vote below
    groups = sorted({v['group'] for v in voyages})
    order = np.random.default_rng(cfg['seed']).permutation(len(groups))
    n_tr, n_va = int(round(cfg['split'][0] * len(groups))), int(round(cfg['split'][1] * len(groups)))
    part = {groups[g]: ('train' if r < n_tr else 'val' if r < n_tr + n_va else 'test') for r, g in enumerate(order)}
    if len(groups) < 3:
        part = {g: 'train' for g in groups}
    for v in voyages:
        v['split'] = part[v['group']]

    # project; SOG unit per voyage (where a voyage cannot tell, the majority of the TRAINING voyages decides,
    # round-4 review finding: the vote must not see validation / test voyages); stale reports
    for v in voyages:
        v['lon0'], v['lat0'] = float(v['lon'][0]), float(v['lat'][0])
        v['xy'] = ll_to_xy(v['lon'], v['lat'], v['lon0'], v['lat0'])
        v['sog_unit'], v['sog_ratio'] = detect_sog_unit(v['t'], v['xy'], v['sog'])
    units = [v['sog_unit'] for v in voyages if v['sog_unit'] and v['split'] == 'train']
    cfg['sog_unit_vote'] = 'training voyages'
    if not units:
        units = [v['sog_unit'] for v in voyages if v['sog_unit']]
        cfg['sog_unit_vote'] = 'ALL voyages (no training voyage could tell)' if units else 'no voyage could tell'
    majority = max(set(units), key=units.count) if units else None
    cfg['sog_unit_majority'] = majority or 'none'
    for i, v in enumerate(voyages):
        unit = v['sog_unit'] or majority
        v['sog_ms'] = None if v['sog'] is None or unit is None else v['sog'] * (KN if unit == 'kn' else 1.0)
        voyages[i] = v = drop_stale(v, counts)
        for k in ('sog', 'cog'):
            if v[k] is not None:
                counts['sog_or_cog_not_available'] += int(np.sum(~np.isfinite(v[k])))
        v['vel_meas'] = (velocity_from_sogcog(v['sog_ms'], v['cog'])
                         if v['sog_ms'] is not None and v['cog'] is not None else None)

    # outlier rules and gap split, per voyage
    segments, dts_all, gaps = [], [], []
    for vi, v in enumerate(voyages):
        dts = np.diff(v['t'])
        dts_all.append(dts)
        auto = min(max(cfg['gap_floor'], cfg['gap_factor'] * float(np.median(dts))), cfg['gap_cap'])
        v['gap_s'] = auto if cfg['gap_s'] == 'auto' else float(cfg['gap_s'])
        v['sparse'] = bool(cfg['gap_factor'] * float(np.median(dts)) > cfg['gap_cap'])
        gaps.append(v['gap_s'])
        if cfg['outlier_rules']:
            bad, cut = distance_outliers(v['t'], v['xy'], v['sog_ms'], cfg['v_cap'], cfg['sigma_p'],
                                         run_fixes=cfg['run_fixes'], run_s=cfg['run_s'])
        else:
            bad, cut = np.zeros(len(v['t']), bool), np.zeros(len(v['t']), bool)
        counts['removed_by_distance_rule'] += int(bad.sum())
        counts['cuts_at_lasting_jumps'] += int(cut.sum())
        v['removed_t'] = list(v['t'][bad])
        segments += _make_segments(vi, v, ~bad, cut, v['gap_s'], counts)
    q0 = 3e-3 if cfg['q'] == 'auto' else float(cfg['q'])
    final = []
    for sg in segments:
        use = residual_outliers(sg['t'], sg['xy'], sg['use'], q0, cfg['sigma_p'], None, cfg['sigma_v'],
                                cfg['gate_floor']) if cfg['outlier_rules'] else sg['use']
        counts['removed_by_residual_pass'] += int(sg['use'].sum() - use.sum())
        voyages[sg['voyage']]['removed_t'] += list(sg['t'][sg['use'] & ~use])
        if use.all():
            final.append(sg)
        else:
            v = dict(t=sg['t'], xy=sg['xy'], sog_ms=sg['sog'], cog=sg['cog'], vel_meas=sg['vel_meas'])
            final += _make_segments(sg['voyage'], v, use, None, voyages[sg['voyage']]['gap_s'], counts)
    segments = final
    counts['segments'] = len(segments)

    # smoothing strength and the COG tests: training voyages only
    train = [sg for sg in segments if voyages[sg['voyage']]['split'] == 'train']
    cfg['q_segments_source'] = 'training voyages'
    if not train:
        # no usable training segment: say so in the report and on the console, then use everything
        # (round-4 review finding: this must never happen silently)
        cfg['q_segments_source'] = ('ALL voyages, validation and test included, because no training voyage '
                                    'left a usable segment')
        print('WARNING: q and the COG tests were estimated on ' + cfg['q_segments_source'])
        train = segments
    if cfg['q'] == 'auto':
        q_used, q_table, q_info = choose_q(train, cfg, rng)
    else:
        _, q_table, q_info = choose_q(train, dict(cfg, q_grid=(q0,), q_bounds=(q0, q0)), rng)
        q_used, q_info['rule'] = q0, 'fixed by the user'
    cfg['q_used'] = q_used
    # SOG / COG enter the smoother only as a receiver velocity that passed the tests; 'on' forces it past them
    aid = bool(use_sogcog is True or (use_sogcog == 'auto' and q_info.get('cog_usable')))
    cfg['aid_forced'] = bool(use_sogcog is True and not q_info.get('cog_usable'))
    if cfg['aid_forced']:
        print('WARNING: --use_sogcog on forces SOG / COG into the smoother although the COG tests did not pass '
              '(tests only; see check_forced_aiding.py)')
    for sg in segments:
        sg['vel'] = sg['vel_meas'] if aid else None

    # smooth on the common absolute grid, cut windows
    W, S, dt = int(cfg['window_points']), int(cfg['stride']), float(cfg['grid_dt'])
    win_xy, win_ll, win_sv, meta = [], [], [], []
    resid = dict(along=[], cross=[], along_end=[], cross_end=[], innov_along=[], innov_cross=[],
                 speed_diff=[], course_diff=[], sd_grid=[], support=[])
    for si, sg in enumerate(segments):
        tu = sg['t'][sg['use']]
        tg = np.arange(math.ceil(tu[0] / dt) * dt, tu[-1] + 1e-9, dt)
        sg['tg'] = tg
        if len(tg) < 2:
            continue
        sm = rts_smooth(sg['t'], sg['xy'], tg, q_used, cfg['sigma_p'], sg['vel'], cfg['sigma_v'], sg['use'])
        sg['grid_pos'], sg['grid_vel'] = sm['pos'], sm['vel']
        _collect_residuals(sg, sm, resid)
        resid['sd_grid'].append(sm['sd'])
        j = np.clip(np.searchsorted(tu, tg), 1, len(tu) - 1)
        resid['support'].append(np.minimum(np.abs(tg - tu[j - 1]), np.abs(tu[j] - tg)) <= dt)
        v = voyages[sg['voyage']]
        for i0 in range(0, len(tg) - W + 1, S):
            pos, vel = sm['pos'][i0:i0 + W], sm['vel'][i0:i0 + W]
            if np.percentile(np.linalg.norm(vel, axis=1), 5) < cfg['min_speed']:
                counts['windows_too_slow'] += 1
                continue
            inside = np.diff(tu[(tu >= tg[i0] - dt) & (tu <= tg[i0 + W - 1] + dt)])
            win_xy.append(pos - pos[0])
            win_ll.append(xy_to_ll(pos, v['lon0'], v['lat0']))
            win_sv.append(np.stack([np.linalg.norm(vel, axis=1), course_deg(vel)], axis=1))
            meta.append(dict(window=len(meta), voyage=sg['voyage'], segment=si, grid_index=i0, group=v['group'],
                             split=v['split'],
                             median_report_interval_s=float(np.median(inside)) if len(inside) else float('nan'),
                             max_report_interval_s=float(np.max(inside)) if len(inside) else float('nan')))
    counts['windows'] = len(meta)

    # optional augmentation of the training windows, inside the envelope of the real training windows
    aug_xy, aug_ll, aug_of, aug_stats = [], [], [], None
    tr_idx = [i for i, m in enumerate(meta) if m['split'] == 'train']
    if augment and tr_idx:
        real = [window_kinematics(win_xy[i], dt) for i in tr_idx]
        lim = dict(v_lo=min(r[0].min() for r in real) * 0.9, v_hi=max(r[0].max() for r in real) * 1.1,
                   acc=1.1 * float(np.percentile([np.abs(r[1]).max() for r in real], 99)),
                   yaw=1.1 * float(np.percentile([np.abs(r[2]).max() for r in real], 99)))
        tried = kept = 0
        for i in tr_idx:
            m, sg = meta[i], segments[meta[i]['segment']]
            for _ in range(int(augment)):
                tried += 1
                w = augment_window(sg, m['grid_index'], cfg, rng)
                sp, ac, yw = window_kinematics(w, dt)
                if sp.min() < lim['v_lo'] or sp.max() > lim['v_hi'] or np.abs(ac).max() > lim['acc'] \
                        or np.abs(yw).max() > lim['yaw']:
                    continue
                kept += 1
                v = voyages[sg['voyage']]
                aug_xy.append(w - w[0]); aug_ll.append(xy_to_ll(w, v['lon0'], v['lat0'])); aug_of.append(m['window'])
        aug_stats = dict(tried=tried, kept=kept, limits_from='training windows',
                         limits={k: round(float(x), 4) for k, x in lim.items()}, real=_kin_quantiles(real),
                         augmented=_kin_quantiles([window_kinematics(w, dt) for w in aug_xy]))

    report = _build_report(cfg, counts, voyages, segments, meta, win_xy, resid, _cat(dts_all), gaps, aid, q_table,
                           q_info, aug_stats, group_key)
    os.makedirs(out_dir, exist_ok=True)
    if meta:
        np.save(os.path.join(out_dir, 'windows_lonlat.npy'), np.asarray(win_ll))      # (n, W, 2) lon, lat
        np.save(os.path.join(out_dir, 'windows_xy.npy'), np.asarray(win_xy))          # metres from the first point
        np.save(os.path.join(out_dir, 'windows_speed_course.npy'), np.asarray(win_sv))
        for name in ('train', 'val', 'test'):
            sel = [i for i, m in enumerate(meta) if m['split'] == name]
            if sel:
                np.save(os.path.join(out_dir, 'windows_lonlat_%s.npy' % name), np.asarray(win_ll)[sel])
        if aug_ll:
            np.save(os.path.join(out_dir, 'augmented_lonlat_train.npy'), np.asarray(aug_ll))
            np.save(os.path.join(out_dir, 'augmented_of_window.npy'), np.asarray(aug_of))
        with open(os.path.join(out_dir, 'windows_meta.csv'), 'w') as fh:      # stays with the data: it names files
            fh.write('window,voyage,segment,grid_index,split,median_report_interval_s,max_report_interval_s,group\n')
            for m in meta:
                fh.write('%d,%d,%d,%d,%s,%.1f,%.1f,%s\n' % (
                    m['window'], m['voyage'], m['segment'], m['grid_index'], m['split'], m['median_report_interval_s'],
                    m['max_report_interval_s'], m['group'].replace(',', ';')))
    with open(os.path.join(out_dir, 'report.json'), 'w') as fh:
        json.dump(report, fh, indent=1)
    with open(os.path.join(out_dir, 'report.md'), 'w') as fh:
        fh.write(report_markdown(report))
    if keep:
        return report, dict(voyages=voyages, segments=segments, meta=meta, cfg=cfg)
    return report


def _kin_quantiles(kin):
    if not kin:
        return {}
    sp = np.concatenate([k[0] for k in kin]); ac = np.abs(np.concatenate([k[1] for k in kin]))
    yw = np.abs(np.concatenate([k[2] for k in kin]))
    q = lambda a: [round(float(np.percentile(a, p)), 4) for p in (50, 95, 99)]
    return dict(speed_p50_p95_p99=q(sp), abs_accel_p50_p95_p99=q(ac), abs_turn_rate_p50_p95_p99=q(yw))


def _collect_residuals(sg, sm, resid):
    use = sg['use']
    res = (sg['xy'] - sm['pos_fix'])[use]
    vel = sm['vel_fix'][use]
    al, cr, ok = track_frame(res, vel)
    tt = sg['t'][use][ok]
    end = (tt - tt[0] < 60.0) | (tt[-1] - tt < 60.0) if len(tt) else np.zeros(0, bool)
    resid['along'].append(al[~end]); resid['cross'].append(cr[~end])
    resid['along_end'].append(al[end]); resid['cross_end'].append(cr[end])
    ia, ic, _ = track_frame(sm['innov'][use], vel)
    resid['innov_along'].append(ia); resid['innov_cross'].append(ic)
    if sg['sog'] is not None:
        good = use & (np.nan_to_num(sg['sog']) > 1.5)
        resid['speed_diff'].append((np.linalg.norm(sm['vel_fix'], axis=1) - sg['sog'])[good])
        if sg['cog'] is not None:
            good = good & np.isfinite(sg['cog'])
            resid['course_diff'].append(wrap180(course_deg(sm['vel_fix']) - sg['cog'])[good])


def _lag1(a):
    a = np.asarray(a, float)
    if len(a) < 10:
        return float('nan')
    a = a - a.mean()
    return float(np.sum(a[1:] * a[:-1]) / max(np.sum(a * a), 1e-12))


def _stats(a):
    a = np.asarray(a, float)
    if not len(a):
        return dict(n=0)
    return dict(n=int(len(a)), median_abs=round(float(np.median(np.abs(a))), 3), rms=round(_rms(a), 3),
                p95_abs=round(_pct(np.abs(a), 95), 3), mean=round(float(a.mean()), 3))


def _q3(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return [round(float(np.percentile(a, p)), 3) for p in (50, 95, 100)] if len(a) else []


def _build_report(cfg, counts, voyages, segments, meta, win_xy, resid, dts, gaps, aid, q_table, q_info, aug_stats,
                  group_key):
    dt = cfg['grid_dt']
    edges = [0, 3, 6, 12, 30, 60, 120, 600, float('inf')]
    hist = np.histogram(dts, bins=edges)[0]
    rep = dict(settings={k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}, counts=counts)
    rep['report_intervals'] = dict(
        median_s=round(float(np.median(dts)), 2), p95_s=round(_pct(dts, 95), 2), histogram_edges_s=[str(e) for e in edges],
        histogram_share=[round(float(h) / max(len(dts), 1), 4) for h in hist],
        gap_threshold_per_voyage_s=_q3(gaps), voyages_too_sparse_for_the_grid=int(sum(v['sparse'] for v in voyages)))
    over = [m['max_report_interval_s'] - voyages[m['voyage']]['gap_s'] for m in meta if np.isfinite(m['max_report_interval_s'])]
    worst = max(over) if over else 0.0
    rep['windows'] = dict(
        longest_interval_minus_threshold_s=round(worst, 1), consistent=bool(worst <= 1e-6),
        median_report_interval_inside_a_window_s=_q3([m['median_report_interval_s'] for m in meta]),
        share_with_median_interval_above_two_grid_steps=round(float(np.mean(
            [m['median_report_interval_s'] > 2 * dt for m in meta])) if meta else float('nan'), 3))
    rep['smoothing'] = dict(q_used=cfg['q_used'], choice=q_info, estimated_on=cfg.get('q_segments_source', 'training voyages'),
                            table=[{k: (round(v, 3) if isinstance(v, float) and k != 'q' else v) for k, v in r.items()}
                                   for r in q_table])
    ratios = [v['sog_ratio'] for v in voyages if np.isfinite(v['sog_ratio'])]
    rep['sog_cog'] = dict(sog_units_detected={str(u): sum(1 for v in voyages if v['sog_unit'] == u) for u in ('kn', 'ms', None)},
                          sog_unit_vote=cfg.get('sog_unit_vote'), sog_unit_majority=cfg.get('sog_unit_majority'),
                          velocity_aiding=bool(aid), aiding_forced=bool(cfg.get('aid_forced')),
                          position_speed_over_sog_p50_p95_max=_q3(ratios),
                          speed_minus_sog_ms=_stats(_cat(resid['speed_diff'])),
                          course_minus_cog_deg=_stats(_cat(resid['course_diff'])))
    ia, ic = _cat(resid['innov_along']), _cat(resid['innov_cross'])
    rep['residuals_m'] = dict(
        along_interior=_stats(_cat(resid['along'])), cross_interior=_stats(_cat(resid['cross'])),
        along_ends=_stats(_cat(resid['along_end'])), cross_ends=_stats(_cat(resid['cross_end'])),
        lag1_autocorr_along=round(float(np.nanmean([_lag1(a) for a in resid['along'] if len(a) > 9] or [np.nan])), 3),
        lag1_autocorr_cross=round(float(np.nanmean([_lag1(a) for a in resid['cross'] if len(a) > 9] or [np.nan])), 3),
        innovation_mean_along=round(float(ia.mean()) if len(ia) else float('nan'), 3),
        innovation_mean_cross=round(float(ic.mean()) if len(ic) else float('nan'), 3))
    sdg, sup = _cat(resid['sd_grid']), _cat(resid['support'])
    rep['grid'] = dict(points=int(len(sdg)), model_position_sd_p50_m=round(_pct(sdg, 50), 2),
                       model_position_sd_p95_m=round(_pct(sdg, 95), 2),
                       share_without_a_report_within_one_step=round(float(1.0 - sup.mean()) if len(sup) else float('nan'), 4))
    geo = dict(fix_to_route=[], fix_to_route_max=[], route_to_fix=[], length_ratio=[], turning=[], kappa=[], yaw=[],
               end_course=[], dev=[], ok=0)
    for m, w in zip(meta, win_xy):
        try:
            r = build_route(w, cfg['route_table_ds'], cfg['route_end_fit'])
        except ValueError:
            continue
        sg = segments[m['segment']]
        t0, t1 = sg['tg'][m['grid_index']], sg['tg'][m['grid_index'] + cfg['window_points'] - 1]
        sel = sg['use'] & (sg['t'] >= t0) & (sg['t'] <= t1)
        raw = sg['xy'][sel] - sg['grid_pos'][m['grid_index']]
        if len(raw) > 3:
            d = perpendicular_distance(raw, r['xy'][::2])
            geo['fix_to_route'].append(float(np.percentile(d, 95))); geo['fix_to_route_max'].append(float(d.max()))
            geo['route_to_fix'].append(float(np.percentile(perpendicular_distance(r['xy'][::10], raw), 95)))
            geo['length_ratio'].append(r['s'][-1] / max(float(np.sum(np.linalg.norm(np.diff(raw, axis=0), axis=1))), 1e-9))
        geo['turning'].append(float(np.degrees(np.sum(np.abs(np.diff(r['psi']))))))
        geo['kappa'].append(float(np.max(np.abs(r['kappa']))))
        geo['yaw'].append(math.degrees(cfg['route_speed'] * geo['kappa'][-1]))
        geo['end_course'].append(float(abs(wrap180(math.degrees(r['course0'] - r['psi'][0])))))
        geo['dev'].append(chord_deviation(w))
        geo['ok'] += int(fit_route_to_limit(w, cfg['route_speed'], cfg['yaw_limit'], 1.0, cfg['route_table_ds'],
                                            cfg['route_end_fit'])[1]['ok'])
    if geo['kappa']:
        rep['routes'] = dict(
            n=len(geo['kappa']), p95_distance_raw_fix_to_route_m=_q3(geo['fix_to_route']),
            largest_distance_raw_fix_to_route_m=_q3(geo['fix_to_route_max']),
            p95_distance_route_to_raw_polyline_m=_q3(geo['route_to_fix']),
            length_route_over_raw_polyline=_q3(geo['length_ratio']), total_turning_deg=_q3(geo['turning']),
            max_curvature_of_the_played_curve_1_per_km=_q3(np.asarray(geo['kappa']) * 1e3),
            playback_speed_ms=cfg['route_speed'], playback_max_yaw_deg_s=_q3(geo['yaw']), yaw_limit_deg_s=cfg['yaw_limit'],
            share_over_yaw_limit=round(float(np.mean(np.asarray(geo['yaw']) > cfg['yaw_limit'])), 3),
            share_usable_after_spatial_smoothing=round(geo['ok'] / len(geo['kappa']), 3),
            start_tangent_against_300m_fit_deg=_q3(geo['end_course']), chord_deviation_m=_q3(geo['dev']),
            share_straight_within_50m=round(float(np.mean(np.asarray(geo['dev']) <= 50.0)), 3))
    parts = {}
    for v in voyages:
        parts.setdefault(v['split'], set()).add(v['group'])
    names = list(parts)
    rep['split'] = dict(grouped_by=group_key, windows={k: sum(1 for m in meta if m['split'] == k) for k in parts},
                        groups={k: len(g) for k, g in parts.items()},
                        overlap_between_splits=bool(any(parts[a] & parts[b] for i, a in enumerate(names) for b in names[i + 1:])))
    if aug_stats:
        rep['augmentation'] = aug_stats
    ext = max(float(np.max(np.linalg.norm(v['xy'], axis=1))) for v in voyages)
    rep['area'] = dict(largest_distance_from_a_voyage_origin_km=round(ext / 1e3, 1),
                       tangent_plane_error_m=round(ext ** 3 / (6 * 6371e3 ** 2), 3))
    return rep


def _fmt(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return '-'
    return '%.2f' % x if isinstance(x, float) else str(x)


def report_markdown(rep):
    L = ['# ais_prep report (aggregates only: no positions, no absolute times, no latitude, no file names)', '']
    L += ['## 1 Counts', '', '| item | n |', '|---|---|'] + ['| %s | %s |' % (k, v) for k, v in rep['counts'].items()] + ['']
    ri, w = rep['report_intervals'], rep['windows']
    L += ['## 2 Report intervals, gaps and windows', '',
          'Median interval %.1f s, 95th percentile %.1f s. Gap threshold per voyage (median, P95, max): %s s. '
          'Voyages too sparse for this grid: %d.' % (ri['median_s'], ri['p95_s'], ri['gap_threshold_per_voyage_s'],
                                                     ri['voyages_too_sparse_for_the_grid']),
          '', '| interval [s] | share |', '|---|---|']
    e = ri['histogram_edges_s']
    L += ['| %s to %s | %.3f |' % (e[i], e[i + 1], s) for i, s in enumerate(ri['histogram_share'])]
    L += ['', 'Consistency: longest report interval inside a window minus its voyage threshold = %.1f s (%s). '
          'Median report interval inside a window (median, P95, max): %s s. Share of windows whose median interval exceeds '
          'two grid steps (their kinematics between reports are interpolated): %s.' % (
              w['longest_interval_minus_threshold_s'], 'consistent' if w['consistent'] else 'INCONSISTENT',
              w['median_report_interval_inside_a_window_s'], w['share_with_median_interval_above_two_grid_steps']), '']
    sm = rep['smoothing']
    ch = sm['choice']
    L += ['## 3 Smoothing strength (estimated on the training voyages)', '',
          'q used: %g m^2/s^3. Rule: %s. Holdout choice %s, COG choice %s, bounds %s, bound active: %s. Estimated on: %s.' % (
              sm['q_used'], ch.get('rule'), ch.get('q_holdout'), ch.get('q_cog'), ch.get('bounds'), ch.get('bound_active'),
              sm.get('estimated_on')),
          'Holdout minimum per voyage (q: number of voyages): %s.' % ch.get('per_voyage_holdout_minimum'),
          'COG: source declared %s; test of independence from the positions %s (difference to the raw fix-to-fix course %s deg RMS, '
          'correlation on straights %s); estimated lag %s s (course against COG by lag: %s); used as a referee and for aiding: %s.' % (
              ch.get('cog_source'), ch.get('cog_independent'), _fmt(ch.get('cog_vs_raw_position_course_rms_deg')),
              _fmt(ch.get('cog_raw_correlation_on_straights')), _fmt(ch.get('cog_lag_s')),
              ch.get('course_vs_cog_rms_by_lag_deg'), ch.get('cog_usable')),
          'Holdout RMS with SOG / COG in the smoother over positions only: %s (has to be at most 1).' % _fmt(
              ch.get('holdout_rms_with_sogcog_over_positions_only')), '',
          '| q | single RMS [m] | single P95 [m] | block RMS [m] | block P95 [m] | n single | n block | '
          'positions-only course vs COG RMS [deg] | speed vs SOG RMS [m/s] |', '|---|---|---|---|---|---|---|---|---|']
    L += ['| %g | %s | %s | %s | %s | %s | %s | %s | %s |' % (
        r['q'], r['single_rms'], r['single_p95'], r['block_rms'], r['block_p95'], r['n_single'], r['n_block'],
        r.get('course_vs_cog_rms_deg', '-'), r.get('speed_vs_sog_rms_ms', '-')) for r in sm['table']] + ['']
    if ch.get('course_vs_cog_by_speed'):
        L += ['Positions-only course minus COG by speed and by turning (turning is read from COG itself):', '',
              '| stratum | n | median abs | RMS | P95 abs | mean |', '|---|---|---|---|---|---|']
        L += ['| %s | %d | %s | %s | %s | %s |' % (k, s['n'], s['median_abs'], s['rms'], s['p95_abs'], s['mean'])
              for k, s in ch['course_vs_cog_by_speed'].items()] + ['']
    sc = rep['sog_cog']
    L += ['## 4 Final track against SOG and COG', '',
          'SOG units detected (voyages): %s; where a voyage cannot tell, the majority (%s) of the %s decides. '
          'Position speed over SOG (median, P95, max over voyages): %s. Velocity aiding: %s.%s%s' % (
              sc['sog_units_detected'], sc.get('sog_unit_majority'), sc.get('sog_unit_vote'),
              sc['position_speed_over_sog_p50_p95_max'], sc['velocity_aiding'],
              ' With aiding on, SOG and COG enter the smoother, so this table only shows that they were followed; the '
              'independent comparison is in section 3.' if sc['velocity_aiding'] else '',
              ' AIDING WAS FORCED (--use_sogcog on) although the COG tests did not pass; tests only.' if sc.get('aiding_forced') else ''), '',
          '| quantity | n | median abs | RMS | P95 abs | mean |', '|---|---|---|---|---|---|']
    for k in ('speed_minus_sog_ms', 'course_minus_cog_deg'):
        if sc[k].get('n'):
            s = sc[k]
            L.append('| %s | %d | %s | %s | %s | %s |' % (k, s['n'], s['median_abs'], s['rms'], s['p95_abs'], s['mean']))
    rs = rep['residuals_m']
    L += ['', '## 5 Residuals, raw fix minus smoothed track [m]', '', '| part | n | median abs | RMS | P95 abs | mean |', '|---|---|---|---|---|---|']
    for k in ('along_interior', 'cross_interior', 'along_ends', 'cross_ends'):
        s = rs[k]
        if s.get('n'):
            L.append('| %s | %d | %s | %s | %s | %s |' % (k, s['n'], s['median_abs'], s['rms'], s['p95_abs'], s['mean']))
    g = rep['grid']
    L += ['', 'Lag-1 autocorrelation: along %.3f, cross %.3f. Mean innovation: along %.3f m, cross %.3f m.' % (
        rs['lag1_autocorr_along'], rs['lag1_autocorr_cross'], rs['innovation_mean_along'], rs['innovation_mean_cross']), '',
          'Grid: %d points; position SD of the model (white errors of sigma_p, not a calibrated uncertainty) median %.1f m, '
          'P95 %.1f m; share without a report within one step %.3f.' % (
              g['points'], g['model_position_sd_p50_m'], g['model_position_sd_p95_m'], g['share_without_a_report_within_one_step']), '']
    if 'routes' in rep:
        L += ['## 6-7 Windows as scenario routes (median, P95, max over windows)', '', '| quantity | values |', '|---|---|']
        L += ['| %s | %s |' % (k, v) for k, v in rep['routes'].items()] + ['']
    sp = rep['split']
    L += ['## 8 Split', '', 'Grouped by %s. Windows %s, groups %s, overlap between splits: %s. One physical voyage stored in '
          'several files under different keys is not detected.' % (sp['grouped_by'], sp['windows'], sp['groups'],
                                                                   sp['overlap_between_splits']), '']
    if 'augmentation' in rep:
        a = rep['augmentation']
        L += ['Augmentation: %d tried, %d kept; limits from the %s: %s.' % (a['tried'], a['kept'], a['limits_from'], a['limits']),
              '', '| set | speed P50/P95/P99 | abs accel | abs turn rate [deg/s] |', '|---|---|---|---|']
        for k in ('real', 'augmented'):
            if a[k]:
                L.append('| %s | %s | %s | %s |' % (k, a[k]['speed_p50_p95_p99'], a[k]['abs_accel_p50_p95_p99'], a[k]['abs_turn_rate_p50_p95_p99']))
        L.append('')
    ar = rep['area']
    L += ['## 9 Area', '', 'Largest distance from a voyage origin %.1f km; tangent-plane error there %.3f m.' % (
        ar['largest_distance_from_a_voyage_origin_km'], ar['tangent_plane_error_m']), '']
    return '\n'.join(L) + '\n'


# ------------------------------------------------------------- legacy check
def legacy_check(paths, n_points=100, columns=None):
    """What the index resampling of data_csv2npy.py did to time: the real time span that each of the
    n_points stands for, from the timestamps alone."""
    rows = []
    for f in collect_csv(paths):
        for tr in read_tracks_csv(f, columns):
            t = tr['t'][np.isfinite(tr['t'])]
            if len(t) < n_points // 4:
                continue
            tk = np.interp(np.linspace(0, len(t) - 1, n_points), np.arange(len(t)), t)
            d = np.diff(tk)
            rows.append(dict(rows=int(len(t)), duration_s=round(float(t[-1] - t[0]), 1), spacing_min_s=round(float(d.min()), 1),
                             spacing_median_s=round(float(np.median(d)), 1), spacing_max_s=round(float(d.max()), 1),
                             max_over_min=round(float(d.max() / max(d.min(), 1e-9)), 1),
                             factor_vs_20s=round(float(np.median(d)) / 20.0, 2), factor_vs_9s=round(float(np.median(d)) / 9.0, 2)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--input', nargs='+', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--pairs', action='store_true', help='CSV files in one folder belong to one encounter: same split')
    p.add_argument('--cog_source', default=None, choices=['unknown', 'receiver'],
                   help="'receiver' when SOG / COG are the ship's own GNSS velocity (decoded AIS messages)")
    p.add_argument('--use_sogcog', default='auto', choices=['auto', 'on', 'off'])
    p.add_argument('--augment', type=int, default=0, help='variants per training window (at most 4)')
    for k in ('grid_dt', 'sigma_p', 'sigma_v', 'gap_floor', 'gap_factor', 'gap_cap', 'min_speed', 'route_speed', 'yaw_limit'):
        p.add_argument('--' + k, type=float, default=None)
    for k in ('window_points', 'stride', 'seed'):
        p.add_argument('--' + k, type=int, default=None)
    p.add_argument('--gap_s', default=None)
    p.add_argument('--q', default=None)
    for k in ALIASES:
        p.add_argument('--col_' + k, default=None, help='column name for %s' % k)
    q = sub.add_parser('legacy-check')
    q.add_argument('--input', nargs='+', required=True)
    for k in ALIASES:
        q.add_argument('--col_' + k, default=None)
    a = ap.parse_args()
    cols = {k: getattr(a, 'col_' + k) for k in ALIASES}
    if a.cmd == 'legacy-check':
        print('| rows | duration [s] | real spacing of the 100 points min / median / max [s] | max/min | median / 20 s | median / 9 s |')
        print('|---|---|---|---|---|---|')
        for r in legacy_check(a.input, columns=cols):
            print('| %d | %.0f | %.1f / %.1f / %.1f | %.1f | %.2f | %.2f |' % (
                r['rows'], r['duration_s'], r['spacing_min_s'], r['spacing_median_s'], r['spacing_max_s'],
                r['max_over_min'], r['factor_vs_20s'], r['factor_vs_9s']))
        return
    over = {k: getattr(a, k) for k in ('grid_dt', 'sigma_p', 'sigma_v', 'gap_floor', 'gap_factor', 'gap_cap', 'min_speed',
                                       'route_speed', 'yaw_limit', 'window_points', 'stride', 'seed', 'cog_source')}
    if a.gap_s is not None:
        over['gap_s'] = a.gap_s if a.gap_s == 'auto' else float(a.gap_s)
    if a.q is not None:
        over['q'] = a.q if a.q == 'auto' else float(a.q)
    rep = prepare(a.input, a.out, columns=cols, pairs=a.pairs,
                  use_sogcog={'auto': 'auto', 'on': True, 'off': False}[a.use_sogcog],
                  augment=min(max(a.augment, 0), 4), **over)
    print(report_markdown(rep))


if __name__ == '__main__':
    main()
