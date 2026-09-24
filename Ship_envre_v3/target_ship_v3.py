"""
Steady target ship (v3): a ship that HOLDS its course and speed on the
approach.

Differences from Ship_envre_v2/target_ship.TargetShip (which it extends):

* steering is course-holding, not pure pursuit to route points: the desired
  course is the local route course at the ship's own projection on the
  route, plus a small cross-track correction (at most `xte_gain_deg` per
  100 m, capped at `xte_cap_deg`).  On a straight route the course is
  therefore constant to the metre; on a curved data route it follows the
  route without cutting corners;
* one alteration per emergency: when the evasion ends the ship returns to
  its route course and a cooldown of `cooldown_steps` decision steps
  (default 30 = 5 min) must pass before it may alter again, so the target
  does not toggle between evading and resuming while the other ship keeps
  closing;
* speed is the cruise speed at all times except the `speed_factor` cut
  while evading (autonomous level), as before.

Routes for the COLREG scenario families are straight by default
(ship_env_v3 passes shape_trace=None); `straighten_route()` turns a
generated data route into its chord when it deviates less than a tolerance.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(os.path.dirname(HERE), 'Ship_envre_v2')
if V2 not in sys.path:
    sys.path.insert(0, V2)

from target_ship import TargetShip, AUTOMATION_LEVELS   # noqa: E402
from encounter import cpa, wrap_pi                      # noqa: E402


def route_deviation(route, length=None):
    """Max distance of the route from its chord [m], over the whole route (default since 2026-09-24,
    mending plan part 1) or over its first `length` metres."""
    route = np.asarray(route, float)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))])
    seg = route if length is None else route[s <= length]
    if len(seg) < 3:
        return 0.0
    chord = seg[-1] - seg[0]
    n = np.array([-chord[1], chord[0]]) / max(np.linalg.norm(chord), 1e-9)
    return float(np.abs((seg - seg[0]) @ n).max())


def straighten_route(route, tol=None, step=100.0):
    """Replace a route by the straight line from its first point along its
    mean course, same length.  With `tol` set, only when the route deviates
    less than `tol` metres from its chord; otherwise the route is returned
    unchanged (its bends are then considered real)."""
    route = np.asarray(route, float)
    if tol is not None and route_deviation(route) > tol:
        return route, False
    length = float(np.sum(np.linalg.norm(np.diff(route, axis=0), axis=1)))
    chord = route[-1] - route[0]
    d = chord / max(np.linalg.norm(chord), 1e-9)
    n = max(int(np.ceil(length / step)), 2)
    pts = route[0] + d[None, :] * np.linspace(0.0, length, n)[:, None]
    return pts, True


class SteadyTargetShip(TargetShip):
    def __init__(self, route_xy, speed, xte_gain_deg=2.0, xte_cap_deg=5.0, cooldown_steps=30, **kw):
        self.xte_gain = np.radians(xte_gain_deg) / 100.0
        self.xte_cap = np.radians(xte_cap_deg)
        self.cooldown_steps = int(cooldown_steps)
        self.cooldown = 0
        self.i_seg = 0
        super().__init__(route_xy, speed, **kw)
        self.cum_s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(self.route, axis=0), axis=1))])

    def reset(self):
        self.cooldown = 0
        self.i_seg = 0
        return super().reset()

    # ---- course holding ---------------------------------------------------
    def _project(self):
        """Segment index and signed cross-track error of the ship on the route."""
        n_seg = len(self.route) - 1
        lo, hi = max(0, self.i_seg - 2), min(n_seg - 1, self.i_seg + 10)
        a = self.route[lo:hi + 1]
        b = self.route[lo + 1:hi + 2]
        ab = b - a
        L2 = np.maximum(np.sum(ab * ab, axis=1), 1e-9)
        t = np.clip(np.sum((self.pos[None, :] - a) * ab, axis=1) / L2, 0.0, 1.0)
        q = a + t[:, None] * ab
        dv = self.pos[None, :] - q
        dist = np.linalg.norm(dv, axis=1)
        j = int(np.argmin(dist))
        self.i_seg = lo + j
        cross = ab[j, 0] * dv[j, 1] - ab[j, 1] * dv[j, 0]
        return self.i_seg, float(np.sign(cross) * dist[j])

    def _route_course(self):
        i, xte = self._project()
        seg = self.route[min(i + 1, len(self.route) - 1)] - self.route[i]
        course = float(np.arctan2(seg[1], seg[0]))
        if i >= len(self.route) - 2:
            self.route_done = True
        # steer back toward the route: positive xte = left of route -> turn right (negative)
        corr = float(np.clip(-self.xte_gain * xte, -self.xte_cap, self.xte_cap))
        return course + corr

    # ---- one alteration per emergency, with cooldown ---------------------
    def _check_emergency(self, dt, p_other, v_other):
        if self.evasion == 'none' or p_other is None:
            return
        if self.cooldown > 0 and not self.evading:
            self.cooldown -= 1
            return
        rng = float(np.linalg.norm(np.asarray(p_other) - self.pos))
        dcpa, tcpa = cpa(self.pos, self.vel, p_other, v_other)
        danger = rng < self.detect_range and dcpa < self.alert_dcpa and 0.0 <= tcpa < self.alert_tcpa
        if danger:
            self.danger_time += dt
            self.clear_count = 0
            if not self.evading and self.danger_time >= self.latency:
                self.evading = True
                self.n_evasions += 1
                if self.evade_mode == 'best':
                    self.evade_offset = self._best_offset(p_other, v_other, self._route_course())
                else:
                    self.evade_offset = -self.evade_angle
        else:
            self.danger_time = 0.0
            if self.evading:
                self.clear_count += 1
                if self.clear_count >= self.clear_steps:
                    self.evading = False
                    self.cooldown = self.cooldown_steps


def steady_target_from_level(level, route_xy, speed, **kw):
    if level not in AUTOMATION_LEVELS:
        raise ValueError('unknown automation level %s' % level)
    params = dict(AUTOMATION_LEVELS[level])
    params.update(kw)
    return SteadyTargetShip(route_xy, speed, automation=level, **params)
