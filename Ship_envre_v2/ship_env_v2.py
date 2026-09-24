"""
ShipAttackEnv: two tasks on generated AIS traces with maritime danger
criteria, COLREG scenario families, target automation levels and an
optional policy-driven other ship (self-play).

  task='attack'    the own ship (attacker) must force a dangerous encounter:
                   severity >= success_severity held for hold_steps steps.
  task='navigate'  the own ship must reach the end of its guideline route
                   (within dest_radius / dest_heading_tol) without ever
                   reaching severity 2 (domain violation) or 3.

Scenario (scenarios.py): 'data' uses the generated merge line as recorded;
'head_on', 'crossing_starboard', 'crossing_port', 'overtaking', 'overtaken'
place the target by the COLREG definition, with the route SHAPE still taken
from a generated sample.  A list means "draw one per episode"
(cycle_scenarios=True cycles instead of drawing, for balanced evaluation).

Other ship (target_ship.py / PolicyShip): automation level 'none',
'manual', 'assisted', 'autonomous' (rule-based, see AUTOMATION_LEVELS) or a
policy callable obs -> action on the full vessel dynamics ('agent').

step(a) -> obs, reward, done, success_flag (1 / 0 / -1), as in
env_moving_attack.MassTestingEnv.  evaluation() returns the episode record
including the event grade / score of grading.py.
"""
import os
import numpy as np

try:
    from gymnasium import spaces as _spaces
except Exception:
    try:
        from gym import spaces as _spaces
    except Exception:
        _spaces = None

from ship_dynamics import ShipDynamics
from encounter import EncounterParams, assess, wrap_pi, SEVERITY_NAMES
from target_ship import TargetShip, smooth_path, route_from_trace, AUTOMATION_LEVELS
from scenarios import SCENARIOS, make_scenario, own_speed_range, colreg_role
from grading import grade_event

EARTH_R = 6371000.0
NM = 1852.0


# ------------------------------------------------------------------ spaces
class _Box:
    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        self.shape = self.low.shape


class _Discrete:
    def __init__(self, n):
        self.n = int(n)
        self.shape = ()


def make_box(low, high):
    if _spaces is not None:
        return _spaces.Box(low=np.asarray(low, dtype=np.float32), high=np.asarray(high, dtype=np.float32),
                           dtype=np.float32)
    return _Box(low, high)


def make_discrete(n):
    return _spaces.Discrete(n) if _spaces is not None else _Discrete(n)


# -------------------------------------------------------------------- data
def load_trace_set(path_or_array):
    """(n, T, 2) array of [lon, lat] samples; a (T, 2) array counts as one sample."""
    if isinstance(path_or_array, (str, os.PathLike)):
        data = np.load(path_or_array, allow_pickle=True)
    else:
        data = np.asarray(path_or_array)
    data = np.asarray(data, dtype=float)
    if data.ndim == 2 and data.shape[1] == 2:
        data = data[None, :, :]
    if data.ndim != 3 or data.shape[2] != 2:
        raise ValueError("trace set must have shape (n, T, 2) or (T, 2), got %s" % (data.shape,))
    return data


def load_trace_speeds(path_or_array, n_samples=None):
    """Median smoothed speed per sample [m/s] of a data_prep set (mending plan part 1, 2026-09-24):
    windows_speed_course.npy (n, T, 2) next to windows_lonlat.npy, the full unsplit array.  None when
    the traces were not given as that path, the file is absent, or the sample counts differ (the split
    arrays windows_lonlat_train / val / test do not align with it row by row)."""
    if not isinstance(path_or_array, (str, os.PathLike)):
        return None
    folder, name = os.path.split(os.fspath(path_or_array))
    if name != 'windows_lonlat.npy':
        return None
    sp = os.path.join(folder, 'windows_speed_course.npy')
    if not os.path.exists(sp):
        return None
    arr = np.asarray(np.load(sp), dtype=float)
    if arr.ndim != 3 or arr.shape[2] < 1 or (n_samples is not None and arr.shape[0] != n_samples):
        return None
    return np.nanmedian(arr[:, :, 0], axis=1)


def lonlat_to_xy(lonlat, lon0, lat0):
    lon = np.radians(lonlat[..., 0])
    lat = np.radians(lonlat[..., 1])
    x = EARTH_R * (lon - np.radians(lon0)) * np.cos(np.radians(lat0))
    y = EARTH_R * (lat - np.radians(lat0))
    return np.stack([x, y], axis=-1)


def cumulative_length(poly):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))])


def project_on_polyline(p, poly, cum_s, i_hint=0, back=2, ahead=10):
    """Arc-length progress, signed cross-track error (positive = left of the
    route) and segment index of the projection of p, searched near i_hint."""
    n_seg = len(poly) - 1
    lo, hi = max(0, i_hint - back), min(n_seg - 1, i_hint + ahead)
    a = poly[lo:hi + 1]
    b = poly[lo + 1:hi + 2]
    ab = b - a
    L2 = np.maximum(np.sum(ab * ab, axis=1), 1e-9)
    t = np.clip(np.sum((p[None, :] - a) * ab, axis=1) / L2, 0.0, 1.0)
    q = a + t[:, None] * ab
    dv = p[None, :] - q
    dist = np.linalg.norm(dv, axis=1)
    j = int(np.argmin(dist))
    i = lo + j
    cross = ab[j, 0] * dv[j, 1] - ab[j, 1] * dv[j, 0]
    s = cum_s[i] + t[j] * (cum_s[i + 1] - cum_s[i])
    return float(s), float(np.sign(cross) * dist[j]), i


OBS_NAMES = ['range', 'cos_brg', 'sin_brg', 'surge', 'sway', 'yaw_rate', 'rel_vx_b', 'rel_vy_b',
             'cos_dpsi', 'sin_dpsi', 't_frac', 'dcpa', 'tcpa', 'severity',
             'dest_range', 'cos_dest_brg', 'sin_dest_brg', 'xte', 'prev_rudder']


def make_obs(state, own_v, other_pos, other_v, other_psi, enc, dest, t_frac, prev_rudder, params,
             d_scale, v_scale, r_scale):
    """The 19-value observation from a ship's own point of view."""
    x, y, psi, u, v, r = state
    rel = other_v - own_v
    c, s = np.cos(psi), np.sin(psi)
    rel_b = np.array([c * rel[0] + s * rel[1], -s * rel[0] + c * rel[1]])
    dpsi = wrap_pi(other_psi - psi)
    tcpa = enc['tcpa'] if np.isfinite(enc['tcpa']) else 5 * params.t_safe
    bearing = wrap_pi(np.arctan2(other_pos[1] - y, other_pos[0] - x) - psi)
    dd = dest - np.array([x, y])
    dest_range = float(np.hypot(dd[0], dd[1]))
    dest_brg = wrap_pi(np.arctan2(dd[1], dd[0]) - psi)
    obs = np.array([
        enc['dist'] / d_scale, np.cos(bearing), np.sin(bearing),
        u / v_scale, v / v_scale, r / r_scale,
        rel_b[0] / v_scale, rel_b[1] / v_scale,
        np.cos(dpsi), np.sin(dpsi), t_frac,
        min(enc['dcpa'] / params.d_safe, 5.0), np.clip(tcpa / params.t_safe, -1.0, 5.0),
        enc['severity'] / 3.0,
        dest_range / d_scale, np.cos(dest_brg), np.sin(dest_brg),
        np.clip(enc['xte'] / 500.0, -5.0, 5.0), prev_rudder,
    ], dtype=np.float32)
    return np.clip(obs, -5, 5)


# ---------------------------------------------------------------- specs
class OwnShipInit:
    def __init__(self, speed_range=(3.0, 10.0), heading_noise_deg=10.0, start_index=0, position_jitter_m=0.0):
        self.speed_range = tuple(speed_range)
        self.heading_noise = np.radians(heading_noise_deg)
        self.start_index = int(start_index)
        self.position_jitter_m = float(position_jitter_m)


class TargetShipSpec:
    def __init__(self, reverse=True, trace_dt=20.0, speed_scale=1.0, speed_scale_range=None,
                 smooth_sigma=2.0, yaw_rate_max_deg=0.5, lookahead=600.0,
                 automation='assisted', overrides=None, clear_steps=6):
        self.reverse = bool(reverse)
        self.trace_dt = float(trace_dt)
        self.speed_scale = float(speed_scale)
        self.speed_scale_range = speed_scale_range
        self.smooth_sigma = float(smooth_sigma)
        self.yaw_rate_max_deg = float(yaw_rate_max_deg)
        self.lookahead = float(lookahead)
        self.automation = automation
        self.overrides = dict(overrides or {})
        self.clear_steps = int(clear_steps)


# ---------------------------------------------------- policy-driven ship
class PolicyShip:
    """The other ship on the full vessel dynamics, steered by a policy
    callable obs -> action (e.g. a trained navigation or attack agent)."""

    def __init__(self, env, route, speed, policy):
        self.env = env
        self.route = np.asarray(route, float)
        self.cum_s = cumulative_length(self.route)
        self.cruise_speed = float(speed)
        self.policy = policy
        self.dyn = ShipDynamics(decision_interval=env.dt)
        self.automation = 'agent'
        self.evasion = 'policy'
        self.reset()

    def reset(self):
        seg = self.route[min(1, len(self.route) - 1)] - self.route[0]
        self.dyn.reset(self.route[0][0], self.route[0][1], float(np.arctan2(seg[1], seg[0])), self.cruise_speed)
        self.i_route = 0
        self.prev_action = np.array([0.0, 0.65])
        self.evading = False
        self.n_evasions = 0
        self.route_done = False
        self.step_count = 0
        self.last_obs = None
        return self.state()

    @property
    def pos(self):
        return self.dyn.state[:2].copy()

    @property
    def psi(self):
        return float(self.dyn.state[2])

    @property
    def vel(self):
        return self.dyn.global_velocity()

    @property
    def speed(self):
        return float(np.linalg.norm(self.vel))

    def state(self):
        return np.array([self.pos[0], self.pos[1], self.psi, self.speed])

    def observe(self, p_other, v_other, psi_other):
        env = self.env
        enc = assess(self.pos, self.vel, self.psi, p_other, v_other, psi_other, env.enc_params)
        s, xte, self.i_route = project_on_polyline(self.pos, self.route, self.cum_s, self.i_route)
        enc['xte'] = xte
        return make_obs(self.dyn.state, self.vel, np.asarray(p_other), np.asarray(v_other), psi_other, enc,
                        self.route[-1], self.step_count / env.max_steps, self.prev_action[0], env.enc_params,
                        env.d_scale, env.v_scale, env.r_scale), enc

    def step(self, dt, p_other=None, v_other=None, psi_other=None):
        obs, enc = self.observe(p_other, v_other, psi_other)
        self.last_obs = obs
        rd, th = self.env._decode_action(self.policy(obs))
        was = self.evading
        self.evading = enc['severity'] >= 1
        if self.evading and not was:
            self.n_evasions += 1
        self.dyn.step(rd, th)
        self.prev_action = np.array([rd, th])
        self.step_count += 1
        return self.state()


# ------------------------------------------------------------------ env
class ShipAttackEnv:
    ACTION_RUDDER = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    ACTION_THROTTLE = np.array([0.3, 0.65, 1.0])
    OBS_NAMES = OBS_NAMES

    def __init__(self, guideline, mergeline, task='attack', own_init=None, target_spec=None, encounter=None,
                 scenario='data', cycle_scenarios=False, other_policy=None,
                 decision_interval=10.0, max_steps=300, border_margin=1500.0,
                 action_mode='discrete', reward_type='shaped',
                 success_severity=2, hold_steps=2, dest_radius=200.0, dest_heading_tol_deg=15.0,
                 d_scale=5000.0, v_scale=15.0, r_scale=0.02,
                 R_success=50.0, R_fail=20.0, R_timeout=10.0, R_danger=30.0, time_cost=0.05,
                 rudder_change_cost=0.05, save_dir=None, seed=None):
        assert task in ('attack', 'navigate')
        self.task = task
        self.own_init = own_init or OwnShipInit()
        self.target_spec = target_spec or TargetShipSpec()
        self.dyn = ShipDynamics(decision_interval=decision_interval)
        self.enc_params = encounter or EncounterParams(L=self.dyn.L)
        self.scenarios = [scenario] if isinstance(scenario, str) else list(scenario)
        if 'mix' in self.scenarios:
            self.scenarios = list(SCENARIOS)
        for sname in self.scenarios:
            assert sname in SCENARIOS, 'unknown scenario %s' % sname
        self.cycle_scenarios = bool(cycle_scenarios)
        self.other_policy = other_policy
        self.dt = float(decision_interval)
        self.max_steps = int(max_steps)
        self.border_margin = float(border_margin)
        self.action_mode = action_mode
        self.reward_type = reward_type
        self.success_severity = int(success_severity)
        self.hold_steps = int(hold_steps)
        self.dest_radius = float(dest_radius)
        self.dest_heading_tol = np.radians(dest_heading_tol_deg)
        self.d_scale, self.v_scale, self.r_scale = d_scale, v_scale, r_scale
        self.R_success, self.R_fail, self.R_timeout, self.R_danger = R_success, R_fail, R_timeout, R_danger
        self.time_cost, self.rudder_change_cost = time_cost, rudder_change_cost
        self.save_dir = save_dir
        self.rng = np.random.default_rng(seed)

        g = load_trace_set(guideline)
        m = load_trace_set(mergeline)
        self.merge_speeds = load_trace_speeds(mergeline, m.shape[0])   # data_prep sets only, else None
        self.lon0 = float(np.mean(g[:, 0, 0]))
        self.lat0 = float(np.mean(g[:, 0, 1]))
        self.guidelines = lonlat_to_xy(g, self.lon0, self.lat0)
        self.mergelines = lonlat_to_xy(m, self.lon0, self.lat0)
        self.n_guide, self.n_merge = self.guidelines.shape[0], self.mergelines.shape[0]
        allpts = np.concatenate([self.guidelines.reshape(-1, 2), self.mergelines.reshape(-1, 2)])
        self.data_bounds = (allpts.min(axis=0), allpts.max(axis=0))
        self._set_borders(allpts)

        self.obs_dim = len(OBS_NAMES)
        self.observation_space = make_box(-np.ones(self.obs_dim) * 5, np.ones(self.obs_dim) * 5)
        if action_mode == 'discrete':
            self.action_table = np.array([[rd, th] for th in self.ACTION_THROTTLE for rd in self.ACTION_RUDDER])
            self.action_space = make_discrete(len(self.action_table))
        else:
            self.action_space = make_box([-1.0, -1.0], [1.0, 1.0])

        self.episode_id = -1
        self.scenario_idx = 0
        self._reset_bookkeeping()

    # ------------------------------------------------------------ helpers
    def seed(self, s=None):
        self.rng = np.random.default_rng(s)
        return [s]

    def _set_borders(self, pts):
        lo = pts.min(axis=0) - self.border_margin
        hi = pts.max(axis=0) + self.border_margin
        self.borders = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [lo[0], hi[1]], [hi[0], hi[1]]])
        self.map_size = hi - lo

    def _reset_bookkeeping(self):
        self.step_count = 0
        self.done = False
        self.success_flag = 0
        self.termination = None
        self.capture_step = None
        self.hold = 0
        self.passed_count = 0
        self.max_severity = 0
        self.sev_steps = {1: 0, 2: 0, 3: 0}
        self.first_encounter = None
        self.psi_at_warning = None
        self.course_change_after_warning = None
        self.own_path, self.target_path, self.target_psi = [], [], []
        self.ownship_action, self.rewards, self.dists = [], [], []
        self.enc_hist, self.evade_hist = [], []
        self.min_distance = np.inf
        self.min_dcpa = np.inf
        self.closing_at_min = 0.0
        self.max_cri = 0.0
        self.prev_action = np.array([0.0, 0.65])

    def _in_borders(self, p):
        return (self.borders[0, 0] <= p[0] <= self.borders[3, 0]) and (self.borders[0, 1] <= p[1] <= self.borders[3, 1])

    def _decode_action(self, a):
        if self.action_mode == 'discrete':
            rd, th = self.action_table[int(a)]
        else:
            a = np.asarray(a, dtype=float).reshape(-1)
            rd = float(np.clip(a[0], -1, 1))
            th = float(0.3 + 0.7 * (np.clip(a[1], -1, 1) + 1) / 2)
        return float(rd), float(th)

    def _assess(self):
        x, y, psi = self.dyn.state[:3]
        own_v = self.dyn.global_velocity()
        enc = assess(np.array([x, y]), own_v, psi, self.target.pos, self.target.vel, self.target.psi, self.enc_params)
        s, xte, self.i_guide = project_on_polyline(np.array([x, y]), self.guide_route, self.guide_s, self.i_guide)
        enc['progress'], enc['xte'] = s, xte
        dd = self.dest - np.array([x, y])
        enc['dest_range'] = float(np.hypot(dd[0], dd[1]))
        enc['dest_bearing'] = float(wrap_pi(np.arctan2(dd[1], dd[0]) - psi))
        enc['bearing'] = float(wrap_pi(np.arctan2(self.target.pos[1] - y, self.target.pos[0] - x) - psi))
        enc['own_speed'] = float(np.linalg.norm(own_v))
        enc['target_speed'] = float(self.target.speed)
        enc['target_evading'] = bool(self.target.evading)
        return enc

    def _observe(self, enc):
        return make_obs(self.dyn.state, self.dyn.global_velocity(), self.target.pos, self.target.vel,
                        self.target.psi, enc, self.dest, self.step_count / self.max_steps, self.prev_action[0],
                        self.enc_params, self.d_scale, self.v_scale, self.r_scale)

    def _build_target(self, route, speed):
        ts = self.target_spec
        if self.other_policy is not None:
            return PolicyShip(self, route, speed, self.other_policy)
        kw = dict(trace_dt=ts.trace_dt, yaw_rate_max_deg=ts.yaw_rate_max_deg, lookahead=ts.lookahead,
                  clear_steps=ts.clear_steps)
        kw.update(ts.overrides)
        return TargetShip.from_level(ts.automation, route, speed, **kw)

    # ----------------------------------------------------------------- API
    def reset(self):
        self.episode_id += 1
        self._reset_bookkeeping()
        ts = self.target_spec
        if self.cycle_scenarios:
            self.scenario = self.scenarios[self.scenario_idx % len(self.scenarios)]
            self.scenario_idx += 1
        else:
            self.scenario = self.scenarios[int(self.rng.integers(len(self.scenarios)))]
        self.colreg_role = colreg_role(self.scenario)

        # own ship on its guideline route
        self.gi = int(self.rng.integers(self.n_guide))
        self.mi = int(self.rng.integers(self.n_merge))
        self.guideline = self.guidelines[self.gi]
        self.guide_route = smooth_path(self.guideline, ts.smooth_sigma)
        self.guide_s = cumulative_length(self.guide_route)
        self.i_guide = 0
        self.dest = self.guide_route[-1].copy()
        seg = self.guide_route[-1] - self.guide_route[-2]
        self.dest_course = float(np.arctan2(seg[1], seg[0]))
        k = self.own_init.start_index
        p0 = self.guide_route[k].copy()
        if self.own_init.position_jitter_m > 0:
            p0 += self.rng.normal(0, self.own_init.position_jitter_m, 2)
        seg = self.guide_route[min(k + 1, len(self.guide_route) - 1)] - self.guide_route[k]
        psi0 = np.arctan2(seg[1], seg[0]) + self.rng.uniform(-self.own_init.heading_noise, self.own_init.heading_noise)
        speed0 = float(self.rng.uniform(*own_speed_range(self.scenario, self.own_init.speed_range)))
        self.init_speed, self.init_heading = speed0, psi0
        self.dyn.reset(p0[0], p0[1], psi0, speed0)

        # target ship route + speed
        merge = self.mergelines[self.mi]
        if ts.speed_scale_range is not None:
            self._speed_scale = float(self.rng.uniform(*ts.speed_scale_range))
        else:
            self._speed_scale = ts.speed_scale
        if self.scenario == 'data':
            self.target_trace = merge[::-1].copy() if ts.reverse else merge.copy()
            route, speed = route_from_trace(self.target_trace, ts.trace_dt, ts.smooth_sigma, self._speed_scale)
            speed_source = 'route length over trace duration'
            if self.merge_speeds is not None and np.isfinite(self.merge_speeds[self.mi]):
                # data_prep set: the median speed of the smoother on this window (mending plan part 1)
                speed = float(self.merge_speeds[self.mi]) * self._speed_scale
                speed_source = 'data_prep windows_speed_course median'
            self.scenario_meta = dict(scenario='data', target_speed=speed, speed_source=speed_source)
        else:
            shape = smooth_path(merge, ts.smooth_sigma)
            sc = make_scenario(self.scenario, p0, psi0, speed0, self.rng, shape_trace=shape)
            route, speed = sc['route'], sc['speed'] * self._speed_scale
            self.target_trace = route
            self.scenario_meta = sc['meta']
        self.target = self._build_target(route, speed)
        self.target.reset()
        self.route_speed = speed
        self._set_borders(np.concatenate([self.guide_route, self.target.route, self.dyn.state[None, :2]]))

        enc = self._assess()
        self._record(enc)
        self.prev_d, self.prev_dcpa, self.prev_s = enc['dist'], self._dcpa_for_shaping(enc), enc['progress']
        return self._observe(enc)

    def _dcpa_for_shaping(self, enc):
        return enc['dcpa'] if np.isfinite(enc['tcpa']) and enc['tcpa'] >= 0 else enc['dist']

    def _record(self, enc):
        self.own_path.append(self.dyn.state[:2].copy())
        self.target_path.append(self.target.pos.copy())
        self.target_psi.append(self.target.psi)
        self.dists.append(enc['dist'])
        self.enc_hist.append(enc)
        self.evade_hist.append(bool(self.target.evading))
        if enc['dist'] < self.min_distance:
            if len(self.dists) > 1:
                self.closing_at_min = (self.dists[-2] - enc['dist']) / self.dt
            self.min_distance = enc['dist']
        if np.isfinite(enc['tcpa']) and enc['tcpa'] >= 0:
            self.min_dcpa = min(self.min_dcpa, enc['dcpa'])
        self.max_cri = max(self.max_cri, enc['cri'])
        sev = enc['severity']
        if sev > 0:
            self.sev_steps[sev] += 1
            if self.first_encounter is None:
                self.first_encounter = (self.step_count, enc['encounter'])
                self.psi_at_warning = float(self.dyn.state[2])
                if self.colreg_role == 'unknown':
                    self.colreg_role = {'head_on': 'both_give_way', 'crossing_starboard': 'give_way',
                                        'crossing_port': 'stand_on', 'overtaking': 'give_way',
                                        'overtaken': 'stand_on'}.get(enc['encounter'], 'unknown')
        if (self.first_encounter is not None and self.course_change_after_warning is None
                and self.step_count - self.first_encounter[0] >= 30):
            self.course_change_after_warning = float(np.degrees(wrap_pi(self.dyn.state[2] - self.psi_at_warning)))
        self.max_severity = max(self.max_severity, sev)

    def step(self, action):
        if self.done:
            raise RuntimeError("step() called after episode end; call reset()")
        rd, th = self._decode_action(action)
        d_rd, d_th = abs(rd - self.prev_action[0]), abs(th - self.prev_action[1])
        self.ownship_action.append([rd, th])
        own_pos_before = self.dyn.state[:2].copy()
        own_vel_before = self.dyn.global_velocity()
        own_psi_before = float(self.dyn.state[2])
        self.dyn.step(rd, th)
        self.target.step(self.dt, own_pos_before, own_vel_before, own_psi_before)
        self.step_count += 1
        self.prev_action = np.array([rd, th])

        enc = self._assess()
        prev_max = max(e['severity'] for e in self.enc_hist)
        self._record(enc)
        sev, d = enc['severity'], enc['dist']
        own_p = self.dyn.state[:2]
        dcpa_s = self._dcpa_for_shaping(enc)

        # ---- termination -------------------------------------------------
        success_flag = 0
        if self.task == 'attack':
            self.hold = self.hold + 1 if sev >= self.success_severity else 0
            if sev == 3 or self.hold >= self.hold_steps:
                self.done, success_flag, self.termination = True, 1, SEVERITY_NAMES[sev]
                self.capture_step = self.step_count
            elif not self._in_borders(own_p):
                self.done, success_flag, self.termination = True, -1, 'own_out_of_map'
            else:
                opening = (enc['tcpa'] < 0 or not np.isfinite(enc['tcpa'])) and d > 2 * self.enc_params.d_safe and d > self.prev_d
                self.passed_count = self.passed_count + 1 if opening else 0
                if self.passed_count >= 6 or (not self._in_borders(self.target.pos) and enc['tcpa'] < 0):
                    self.done, success_flag, self.termination = True, -1, 'passed'
                elif self.step_count >= self.max_steps:
                    self.done, success_flag, self.termination = True, -1, 'timeout'
        else:
            reached = enc['dest_range'] < self.dest_radius and abs(wrap_pi(self.dyn.state[2] - self.dest_course)) < self.dest_heading_tol
            if sev >= 2:
                self.done, success_flag, self.termination = True, -1, SEVERITY_NAMES[sev]
            elif reached:
                self.done, success_flag, self.termination = True, 1, 'destination'
                self.capture_step = self.step_count
            elif not self._in_borders(own_p):
                self.done, success_flag, self.termination = True, -1, 'own_out_of_map'
            elif self.step_count >= self.max_steps:
                self.done, success_flag, self.termination = True, -1, 'timeout'

        # ---- reward ------------------------------------------------------
        manoeuvre_cost = self.rudder_change_cost * (d_rd + 0.5 * d_th)
        if self.reward_type == 'final_attack_reward':
            rew = -self.time_cost - manoeuvre_cost
        elif self.task == 'attack':
            rew = ((self.prev_d - d) / 100.0 + float(np.clip((self.prev_dcpa - dcpa_s) / 200.0, -2.0, 2.0))
                   + 0.1 * np.cos(enc['bearing']) - self.time_cost - manoeuvre_cost)
            if sev > prev_max:
                rew += {1: 5.0, 2: 15.0, 3: 15.0}[sev]
        else:
            rew = ((enc['progress'] - self.prev_s) / 100.0 - 0.5 * min(abs(enc['xte']), 1000.0) / 1000.0
                   - self.time_cost - manoeuvre_cost - (1.0 if sev == 1 else 0.0))
        if self.termination in ('cpa_warning', 'domain_violation', 'collision'):
            rew += self.R_success if self.task == 'attack' else -(self.R_danger + (20.0 if sev == 3 else 0.0))
        elif self.termination == 'destination':
            rew += self.R_success
        elif self.termination == 'own_out_of_map':
            rew -= self.R_fail
        elif self.termination in ('timeout', 'passed'):
            rew -= self.R_timeout
        self.prev_d, self.prev_dcpa, self.prev_s = d, dcpa_s, enc['progress']
        self.rewards.append(float(rew))
        self.success_flag = success_flag
        self.last_info = dict(enc, termination=self.termination)
        return self._observe(enc), float(rew), self.done, success_flag

    @property
    def destination_step(self):
        return self.capture_step if self.capture_step is not None else self.step_count

    # ------------------------------------------------------------ analysis
    def evaluation(self):
        own = np.asarray(self.own_path)
        act = np.asarray(self.ownship_action) if self.ownship_action else np.zeros((0, 2))
        path_len = float(np.sum(np.linalg.norm(np.diff(own, axis=0), axis=1))) if len(own) > 1 else 0.0
        xte = [abs(e['xte']) for e in self.enc_hist]
        d_act = np.abs(np.diff(act, axis=0)) if len(act) > 1 else np.zeros((1, 2))
        ev = dict(
            episode=self.episode_id, task=self.task, scenario=self.scenario,
            automation=getattr(self.target, 'automation', 'custom'), colreg_role=self.colreg_role,
            success=int(self.success_flag == 1), termination=self.termination,
            steps=self.step_count, dt=self.dt,
            capture_step=self.capture_step if self.capture_step is not None else -1,
            destination_step=self.destination_step,
            max_severity=self.max_severity, steps_sev1=self.sev_steps[1], steps_sev2=self.sev_steps[2],
            steps_sev3=self.sev_steps[3],
            min_distance=float(self.min_distance), min_dcpa=float(self.min_dcpa if np.isfinite(self.min_dcpa) else -1),
            closing_speed_at_min=float(self.closing_at_min), max_cri=float(self.max_cri),
            first_encounter_step=self.first_encounter[0] if self.first_encounter else -1,
            first_encounter_type=self.first_encounter[1] if self.first_encounter else 'none',
            course_change_after_warning_deg=self.course_change_after_warning,
            target_evasions=int(self.target.n_evasions), target_speed=float(self.route_speed),
            final_distance=float(self.dists[-1]), own_path_len=path_len,
            mean_speed=path_len / max(self.step_count * self.dt, 1e-9),
            rudder_mean_abs=float(np.mean(np.abs(act[:, 0]))) if len(act) else 0.0,
            rudder_change_mean=float(np.mean(d_act[:, 0])), throttle_change_mean=float(np.mean(d_act[:, 1])),
            xte_mean=float(np.mean(xte)), progress=float(self.enc_hist[-1]['progress']),
            route_length=float(self.guide_s[-1]), dest_range_final=float(self.enc_hist[-1]['dest_range']),
            total_reward=float(np.sum(self.rewards)),
            guide_sample=self.gi, merge_sample=self.mi,
            init_speed=float(self.init_speed), init_heading_deg=float(np.degrees(self.init_heading)),
        )
        for k_, v_ in self.scenario_meta.items():
            if k_ not in ('scenario', 'target_speed'):
                ev['scn_' + k_] = v_
        grade, gname, score = grade_event(ev, self.enc_params)
        ev.update(event_grade=grade, event_grade_name=gname, event_score=score)
        return ev

    def show_scenes(self, save_path=None, title=None, show=False):
        import matplotlib
        if not show:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse, Circle
        own = np.asarray(self.own_path)
        tgt = np.asarray(self.target_path)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(self.guide_route[:, 0], self.guide_route[:, 1], color='0.6', lw=1, ls='--', label='guideline route')
        ax.plot(self.target.route[:, 0], self.target.route[:, 1], color='salmon', lw=1, ls='--', label='target route')
        ax.plot(own[:, 0], own[:, 1], 'b-', lw=2, label='own ship')
        ax.plot(tgt[:, 0], tgt[:, 1], 'r-', lw=2, label='target ship')
        ev = np.asarray(self.evade_hist, bool)
        if ev.any():
            ax.plot(tgt[ev, 0], tgt[ev, 1], 'o', color='orange', ms=3, label='target evading')
        ax.plot(own[0, 0], own[0, 1], 'bo'); ax.plot(tgt[0, 0], tgt[0, 1], 'ro')
        ax.plot(own[-1, 0], own[-1, 1], 'bs'); ax.plot(tgt[-1, 0], tgt[-1, 1], 'rs')
        ax.add_patch(Ellipse(tgt[-1], 2 * self.enc_params.domain_a, 2 * self.enc_params.domain_b,
                             angle=np.degrees(self.target_psi[-1]), fc='none', ec='r', ls=':', lw=1))
        ax.add_patch(Circle(tgt[-1], self.enc_params.d_safe, fc='none', ec='orange', ls=':', lw=0.7))
        if self.task == 'navigate':
            ax.add_patch(Circle(self.dest, self.dest_radius, fc='none', ec='g', ls='-', lw=1))
        bx = self.borders
        ax.plot([bx[0, 0], bx[1, 0], bx[3, 0], bx[2, 0], bx[0, 0]],
                [bx[0, 1], bx[1, 1], bx[3, 1], bx[2, 1], bx[0, 1]], 'k:', lw=0.8)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
        ax.set_title(title or '%s / %s ep %d: %s, %d steps, max severity %d, min DCPA %.0f m' % (
            self.task, self.scenario, self.episode_id, self.termination, self.step_count, self.max_severity,
            self.min_dcpa if np.isfinite(self.min_dcpa) else -1))
        ax.legend(loc='best', fontsize=8)
        if save_path is None and self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            save_path = os.path.join(self.save_dir, 'scene_episode_%d.png' % self.episode_id)
        if save_path is not None:
            fig.savefig(save_path, dpi=120, bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)
        return save_path

    def close(self):
        pass


# ------------------------------------------------------- scripted policies
def _discrete_from(env, rd, th):
    ri = int(np.argmin(np.abs(env.ACTION_RUDDER - rd)))
    ti = int(np.argmin(np.abs(env.ACTION_THROTTLE - th)))
    return ti * len(env.ACTION_RUDDER) + ri


def pursuit_policy(env, obs):
    """Attack baseline: steer to the target's bearing, full throttle."""
    bearing = np.arctan2(obs[2], obs[1])
    rd = float(np.clip(bearing / np.radians(20), -1, 1))
    if env.action_mode == 'discrete':
        return _discrete_from(env, rd, 1.0)
    return np.array([rd, 1.0])


def guideline_policy(env, obs):
    """Navigate baseline: pure pursuit on the guideline route, mid throttle,
    starboard turn while the CPA warning is active (COLREG Rule 8/14/15)."""
    x, y, psi = env.dyn.state[:3]
    route = env.guide_route
    i = env.i_guide
    while i < len(route) - 1 and np.linalg.norm(route[i] - np.array([x, y])) < 600.0:
        i += 1
    d = route[i] - np.array([x, y])
    desired = np.arctan2(d[1], d[0])
    enc = env.enc_hist[-1]
    if enc['severity'] >= 1 and enc['bearing'] > -np.radians(112.5):
        desired -= np.radians(35)
    err = wrap_pi(desired - psi)
    rd = float(np.clip(err / np.radians(15), -1, 1))
    th = 0.65
    if env.action_mode == 'discrete':
        return _discrete_from(env, rd, th)
    return np.array([rd, (th - 0.3) / 0.7 * 2 - 1])


if __name__ == '__main__':
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    from make_synthetic_data import make_synthetic_traces
    g, m = make_synthetic_traces(n=50, seed=0)
    for task, policy in [('attack', pursuit_policy), ('navigate', guideline_policy)]:
        for level in ['manual', 'autonomous']:
            env = ShipAttackEnv(g, m, task=task, scenario='mix', cycle_scenarios=True,
                                target_spec=TargetShipSpec(automation=level), seed=1)
            res = []
            for ep in range(len(SCENARIOS)):
                obs = env.reset(); done = False
                while not done:
                    obs, r, done, flag = env.step(policy(env, obs))
                e = env.evaluation()
                res.append('%s:%s/%d/g%d/%.0f' % (e['scenario'][:8], e['termination'][:8], e['steps'],
                                                 e['event_grade'], e['event_score']))
            print(task, level, res)
