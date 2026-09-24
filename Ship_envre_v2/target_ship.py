"""
Ship-like target model with automation levels.

The generated merge-line trace (or a scenario route from scenarios.py) is a
smooth ROUTE.  The target follows it the way a ship does: constant speed,
course held, turn rate limited, steering to a look-ahead point (pure
pursuit).  Course and speed only change when an emergency criterion is met
from the target's own point of view:

    range < detect_range  and  DCPA < alert_dcpa  and  0 <= TCPA < alert_tcpa

held for `latency` seconds (reaction time).  Then it performs a COLREG-style
evasive manoeuvre - alter course to starboard (Rules 8, 14-17) by a fixed
angle, or by the candidate angle that maximises the predicted DCPA
(evade_mode='best'), optionally reducing speed - and holds it until the
approach is clear for `clear_steps` consecutive checks, after which pure
pursuit brings it back onto the route at cruise speed.

Automation levels (stand-in parameterisations of the IMO MASS degrees):

  none        passive, never reacts                       (no bridge team)
  manual      conventional bridge team: 90 s reaction, alert at 0.5 nm / 12 min,
              30 deg fixed alteration, 6 nm detection         (MASS degree 0-1)
  assisted    decision support / remote operator: 30 s, 1 nm / 15 min, 40 deg,
              8 nm                                            (MASS degree 2)
  autonomous  collision-avoidance algorithm: 0 s, 1 nm / 20 min, best of
              20/40/60 deg, speed cut to 70 %, 10 nm          (MASS degree 3-4)
  agent       policy-driven ship on the full dynamics, see ship_env_v2.PolicyShip

Speed comes from the trace itself unless a scenario sets it.
"""
import numpy as np
from encounter import cpa, wrap_pi

NM = 1852.0

AUTOMATION_LEVELS = {
    'none':       dict(evasion='none'),
    'manual':     dict(evasion='colreg', latency=90.0, alert_dcpa=0.5 * NM, alert_tcpa=720.0,
                       evade_mode='fixed', evade_angle_deg=30.0, speed_factor=1.0, detect_range=6 * NM),
    'assisted':   dict(evasion='colreg', latency=30.0, alert_dcpa=1.0 * NM, alert_tcpa=900.0,
                       evade_mode='fixed', evade_angle_deg=40.0, speed_factor=1.0, detect_range=8 * NM),
    'autonomous': dict(evasion='colreg', latency=0.0, alert_dcpa=1.0 * NM, alert_tcpa=1200.0,
                       evade_mode='best', evade_angle_deg=60.0, speed_factor=0.7, detect_range=10 * NM),
}


def smooth_path(xy, sigma_points=2.0):
    """Gaussian smoothing along the trace (in metres) with edge padding."""
    if sigma_points <= 0:
        return np.asarray(xy, float).copy()
    xy = np.asarray(xy, float)
    half = int(np.ceil(3 * sigma_points))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma_points) ** 2)
    k /= k.sum()
    pad = np.concatenate([np.repeat(xy[:1], half, 0), xy, np.repeat(xy[-1:], half, 0)])
    return np.stack([np.convolve(pad[:, i], k, mode='valid') for i in range(2)], 1)


def path_length(xy):
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))


class TargetShip:
    def __init__(self, route_xy, speed, trace_dt=20.0, yaw_rate_max_deg=0.5, steer_gain=0.1,
                 lookahead=600.0, evasion='colreg', evade_angle_deg=40.0,
                 alert_dcpa=1852.0, alert_tcpa=900.0, clear_steps=6, substep=1.0,
                 latency=0.0, evade_mode='fixed', speed_factor=1.0, detect_range=10 * NM,
                 automation='custom'):
        self.route = np.asarray(route_xy, float)
        self.cruise_speed = float(speed)
        self.speed = float(speed)
        self.r_max = np.radians(yaw_rate_max_deg)
        self.k = float(steer_gain)
        self.lookahead = float(lookahead)
        self.evasion = evasion
        self.evade_angle = np.radians(evade_angle_deg)
        self.alert_dcpa, self.alert_tcpa = float(alert_dcpa), float(alert_tcpa)
        self.clear_steps = int(clear_steps)
        self.h = float(substep)
        self.latency = float(latency)
        self.evade_mode = evade_mode
        self.speed_factor = float(speed_factor)
        self.detect_range = float(detect_range)
        self.automation = automation
        self.reset()

    @classmethod
    def from_level(cls, level, route_xy, speed, **kw):
        """Build a target for an automation level name; kw overrides."""
        if level not in AUTOMATION_LEVELS:
            raise ValueError('unknown automation level %s (choose %s)' % (level, list(AUTOMATION_LEVELS)))
        params = dict(AUTOMATION_LEVELS[level])
        params.update(kw)
        return cls(route_xy, speed, automation=level, **params)

    # ------------------------------------------------------------------
    def reset(self):
        self.pos = self.route[0].copy()
        seg = self.route[min(1, len(self.route) - 1)] - self.route[0]
        self.psi = float(np.arctan2(seg[1], seg[0]))
        self.speed = self.cruise_speed
        self.i_target = 1
        self.evading = False
        self.danger_time = 0.0
        self.clear_count = 0
        self.n_evasions = 0
        self.route_done = False
        self.evade_offset = -self.evade_angle
        self.t = 0.0
        return self.state()

    def state(self):
        return np.array([self.pos[0], self.pos[1], self.psi, self.speed])

    @property
    def vel(self):
        return self.speed * np.array([np.cos(self.psi), np.sin(self.psi)])

    # ------------------------------------------------------------------
    def _route_course(self):
        n = len(self.route)
        while self.i_target < n - 1 and np.linalg.norm(self.route[self.i_target] - self.pos) < self.lookahead:
            self.i_target += 1
        tgt = self.route[self.i_target]
        if self.i_target >= n - 1 and np.linalg.norm(tgt - self.pos) < self.lookahead:
            self.route_done = True
            seg = self.route[-1] - self.route[-2]
            return float(np.arctan2(seg[1], seg[0]))
        d = tgt - self.pos
        return float(np.arctan2(d[1], d[0]))

    def _best_offset(self, p_other, v_other, route_course):
        """Starboard alteration among 20/40/60 deg that maximises predicted DCPA."""
        best, best_d = -self.evade_angle, -1.0
        for deg in (20.0, 40.0, 60.0):
            off = -np.radians(deg)
            psi = route_course + off
            vel = self.cruise_speed * self.speed_factor * np.array([np.cos(psi), np.sin(psi)])
            d, t = cpa(self.pos, vel, p_other, v_other)
            if t < 0:
                d = float(np.linalg.norm(np.asarray(p_other) - self.pos))
            if d > best_d:
                best, best_d = off, d
        return best

    def _check_emergency(self, dt, p_other, v_other):
        if self.evasion == 'none' or p_other is None:
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

    def step(self, dt, p_other=None, v_other=None, psi_other=None):
        """Advance dt seconds; emergency assessed once per call."""
        self._check_emergency(dt, p_other, v_other)
        target_speed = self.cruise_speed * (self.speed_factor if self.evading else 1.0)
        n_sub = max(1, int(round(dt / self.h)))
        h = dt / n_sub
        for _ in range(n_sub):
            desired = self._route_course()
            if self.evading:
                desired = desired + self.evade_offset
            err = wrap_pi(desired - self.psi)
            r = float(np.clip(self.k * err, -self.r_max, self.r_max))
            self.psi = wrap_pi(self.psi + r * h)
            # speed changes slowly (large ship): 0.01 m/s per second
            self.speed += float(np.clip(target_speed - self.speed, -0.01 * h, 0.01 * h))
            self.pos = self.pos + self.vel * h
            self.t += h
        return self.state()


def route_from_trace(trace_xy, trace_dt, smooth_sigma=2.0, speed_scale=1.0):
    """Smooth a generated trace into a route and derive the cruise speed."""
    route = smooth_path(trace_xy, smooth_sigma)
    duration = (len(trace_xy) - 1) * trace_dt
    speed = path_length(route) / max(duration, 1e-9) * speed_scale
    return route, speed
