"""Probe the current trace sampling + smoothing chain on a synthetic AIS-like track
whose truth is known (the real AIS data are private, so only the code is tested).

Chain under test (as in the code):
  S1 data_csv2npy.interpolation_downsample: cubic spline over ROW INDEX -> 100 pts
  S2 data_csv2npy augmentation: tsgm GaussianNoise variance 1e-4 on degrees (sigma = sqrt(var))
  S3 Ship_envre/interpolation.py: assumes 9 s spacing, cubic -> 1 s -> 20 s
  S4 PPO_scenario_generate.load_trace_shapes: equirectangular to metres + smooth_path (box window 5)
  S5 v2 target_ship.smooth_path (Gaussian sigma 2 pts) + route_from_trace speed = length / ((n-1) dt)
  S6 rotate_translate initial course from point 3 - point 0
Truth: 60 min voyage at 57.6 N, 6 m/s then 4 m/s, one 90 deg turn at 0.3 deg/s,
AIS reports every 2-10 s while turning/fast plus 3 gaps of 2-6 min, GPS noise 5 m.
"""
import math, sys, numpy as np
from scipy import interpolate
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'general'))   # the general folder (was PPO_scenario_generate)
from scenario_targets import smooth_path as box_smooth, rotate_translate

R = 6371000.0
LAT0, LON0 = 57.6, 11.9
rng = np.random.default_rng(1)


def truth(t):
    """Continuous truth: x,y [m], speed, heading [deg math]."""
    x = y = 0.0; psi = 30.0
    out = []; tt = 0.0; dt = 0.5
    ts = np.arange(0, t[-1] + dt, dt)
    for tk in ts:
        v = 6.0 if tk < 1800 else 4.0
        if 1500 <= tk < 1800:
            psi += 0.3 * dt
        x += v * math.cos(math.radians(psi)) * dt; y += v * math.sin(math.radians(psi)) * dt
        out.append((x, y, v, psi))
    out = np.array(out)
    return np.column_stack([np.interp(t, ts, out[:, i]) for i in range(4)])


def xy2ll(xy):
    lat = LAT0 + np.degrees(xy[:, 1] / R)
    lon = LON0 + np.degrees(xy[:, 0] / (R * math.cos(math.radians(LAT0))))
    return np.column_stack([lon, lat])


def ll2xy(ll):
    return np.column_stack([np.radians(ll[:, 0] - LON0) * R * math.cos(math.radians(LAT0)),
                            np.radians(ll[:, 1] - LAT0) * R])


# ---- AIS-like irregular sampling
t = [0.0]
while t[-1] < 3600:
    t.append(t[-1] + rng.uniform(2, 10))
t = np.array(t)
for g0, g in [(600, 180), (2200, 360), (3000, 120)]:             # reception gaps
    t = t[(t < g0) | (t > g0 + g)]
T = truth(t)
ais_xy = T[:, :2] + rng.normal(0, 5.0, (len(t), 2))
ais_ll = xy2ll(ais_xy)
print(f"AIS-like input: {len(t)} reports over {t[-1]/60:.0f} min, spacing 2-10 s + gaps 120-360 s")


def spline_index(data, n=100):                                    # S1, verbatim logic
    u = np.arange(len(data))
    tck, _ = interpolate.splprep([data[:, i] for i in range(data.shape[1])], u=u, s=0)
    return np.column_stack(interpolate.splev(np.linspace(0, len(data) - 1, n), tck))


def err_vs_truth(xy_s, t_s):
    tr = truth(t_s)[:, :2]
    return np.linalg.norm(xy_s - tr, axis=1)


def speeds(xy, dt):
    return np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt


def max_turn_rate(xy, dt):
    h = np.unwrap(np.arctan2(np.diff(xy[:, 1]), np.diff(xy[:, 0])))
    return np.degrees(np.abs(np.diff(h))).max() / dt


# S1: index-spline to 100 points, then (as the chain does) treat points as uniform in time
s1_ll = spline_index(ais_ll)
s1_xy = ll2xy(s1_ll)
u_idx = np.linspace(0, len(t) - 1, 100)
t_true_of_pt = np.interp(u_idx, np.arange(len(t)), t)             # real time each point stands for
t_uniform = np.linspace(0, t[-1], 100)                            # what downstream assumes (dt = T/99)
dt_s1 = t[-1] / 99
print("\nS1 cubic spline over row index -> 100 points")
print(f"  real time between consecutive points: {np.diff(t_true_of_pt).min():.1f} - {np.diff(t_true_of_pt).max():.1f} s "
      f"(downstream assumes a constant {dt_s1:.1f} s)")
e_time = np.abs(t_true_of_pt - t_uniform)
print(f"  timing error of a point if read as uniform: median {np.median(e_time):.0f} s, max {e_time.max():.0f} s")
v_s1 = speeds(s1_xy, dt_s1)
print(f"  implied speed from uniform-time reading: {v_s1.min():.2f} - {v_s1.max():.2f} m/s (truth 4.00 / 6.00)")
# spline shape error vs truth at the true times
e1 = err_vs_truth(s1_xy, t_true_of_pt)
print(f"  position error vs truth at the point's true time: median {np.median(e1):.1f} m, max {e1.max():.1f} m")
# time-based alternative
t_grid = np.arange(0, t[-1], 20.0)
alt = np.column_stack([np.interp(t_grid, t, ais_xy[:, i]) for i in range(2)])
alt_s = box_smooth(alt, 3)
ea = err_vs_truth(alt_s, t_grid)
print(f"  [alternative] linear on timestamps at 20 s + window 3: position error median {np.median(ea):.1f} m, "
      f"max {ea.max():.1f} m, speed {np.percentile(speeds(alt_s, 20), 5):.2f}-{np.percentile(speeds(alt_s, 20), 95):.2f} m/s (p5-p95)")

# S2: Gaussian noise augmentation on degrees
for var in (1e-4, 1e-8):
    sig = math.sqrt(var)
    noisy = s1_ll + rng.normal(0, sig, s1_ll.shape)
    nxy = ll2xy(noisy)
    d = np.linalg.norm(nxy - s1_xy, axis=1)
    v = speeds(nxy, dt_s1)
    print(f"\nS2 GaussianNoise variance {var:g} (sigma {sig:g} deg)")
    print(f"  displacement per point: median {np.median(d):.0f} m (lat {sig*R*math.pi/180:.0f} m, lon {sig*R*math.pi/180*math.cos(math.radians(LAT0)):.0f} m)")
    print(f"  implied speed: median {np.median(v):.1f} m/s, max {v.max():.1f} m/s")
    for name, sm in (('box 5 (PPO_scenario_generate)', lambda a: box_smooth(a, 5)),
                     ('Gaussian 2 (v2/v3)', None)):
        if sm is None:
            h = 6; k = np.exp(-0.5 * (np.arange(-h, h + 1) / 2.0) ** 2); k /= k.sum()
            pad = np.concatenate([np.repeat(nxy[:1], h, 0), nxy, np.repeat(nxy[-1:], h, 0)])
            sxy = np.stack([np.convolve(pad[:, i], k, 'valid') for i in range(2)], 1)
        else:
            sxy = sm(nxy)
        e = np.linalg.norm(sxy - s1_xy, axis=1)
        L = np.sum(np.linalg.norm(np.diff(sxy, axis=0), axis=1)); L0 = np.sum(np.linalg.norm(np.diff(s1_xy, axis=0), axis=1))
        print(f"  after {name}: residual error median {np.median(e):.0f} m; path length x{L/L0:.2f} "
              f"-> v2 route_from_trace speed x{L/L0:.2f}; max turn rate {max_turn_rate(sxy, dt_s1):.2f} deg/s")
        ang = math.degrees(math.atan2(*(sxy[3] - sxy[0])[::-1])) - math.degrees(math.atan2(*(s1_xy[3] - s1_xy[0])[::-1]))
        print(f"    S6 initial course (point 3 - point 0) error: {((ang + 180) % 360) - 180:+.1f} deg")

# S4 / S5 on clean data: what smoothing alone does to a real turn
print("\nS4/S5 smoothing alone on the clean 100-point trace (turn 90 deg at 0.3 deg/s)")
Lc = np.sum(np.linalg.norm(np.diff(s1_xy, axis=0), axis=1))
for name, sxy in (('box 5', box_smooth(s1_xy, 5)),):
    e = np.linalg.norm(sxy - s1_xy, axis=1)
    L = np.sum(np.linalg.norm(np.diff(sxy, axis=0), axis=1))
    print(f"  {name}: max offset {e.max():.1f} m (end points {e[0]:.1f}/{e[-1]:.1f} m), length x{L/Lc:.4f}, "
          f"turn rate max {max_turn_rate(sxy, dt_s1):.3f} vs {max_turn_rate(s1_xy, dt_s1):.3f} deg/s")

# S3 interpolation.py: assumes 9 s spacing
print("\nS3 Ship_envre/interpolation.py assumes 9 s between points")
print(f"  the 100-point S1 output spans {t[-1]:.0f} s -> real spacing {dt_s1:.1f} s; read as 9 s the voyage "
      f"is compressed x{dt_s1/9:.1f} in time and speeds are inflated x{dt_s1/9:.1f}")

# where does box smoothing move the clean trace most?
sxy = box_smooth(s1_xy, 5)
e = np.linalg.norm(sxy - s1_xy, axis=1)
seg = np.linalg.norm(np.diff(s1_xy, axis=0), axis=1)
i = int(np.argmax(e))
print(f"\nlargest box-5 shift at point {i}: neighbouring segment lengths "
      f"{np.round(seg[max(i-3,0):i+3]).astype(int).tolist()} m (median segment {np.median(seg):.0f} m)")
top = np.argsort(e)[-5:]
print("  top-5 shifted points:", sorted(top.tolist()), "shifts", np.round(np.sort(e)[-5:]).astype(int).tolist())
# arc-length resampled first, then smoothed
s = np.concatenate([[0], np.cumsum(seg)])
g = np.arange(0, s[-1], np.median(seg))
ar = np.column_stack([np.interp(g, s, s1_xy[:, k]) for k in range(2)])
ea = np.linalg.norm(box_smooth(ar, 5) - ar, axis=1)
print(f"  [alternative] resample by arc length first, then box 5: max shift {ea.max():.1f} m, interior max {ea[3:-3].max():.1f} m")
