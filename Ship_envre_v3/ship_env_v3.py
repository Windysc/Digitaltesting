"""
ShipEncounterEnv (v3): the encounter-lifecycle definition.

No ship "ends" at a point or at a time.  Both ships sail open-ended routes
(the generated routes, extended along their final course) and an EPISODE is
one ENCOUNTER, delimited by its own lifecycle:

    phase 0  approach   CPA ahead, no warning yet
    phase 1  action     a CPA warning or higher severity is / was active
    phase 2  passing    CPA behind (TCPA < 0)
    phase 3  clear      no CPA criterion met, range above r_clear and
                        opening for n_clear consecutive steps
                        (if the other ship turns back and closes again the
                        count resets, so a second approach continues the
                        same encounter)

Outcomes
  task='attack'    success (+1): severity >= success_severity held for
                   hold_steps, or collision.  Failure (-1): the encounter
                   reaches "clear" without the event, or the ship leaves the
                   map.  No timeout failure.
  task='navigate'  success (+1): the encounter reaches "clear" with maximum
                   severity <= 1 AND the ship has recovered its passage plan
                   (|cross-track error| < xte_tol and course within
                   course_tol of the route for recover_steps consecutive
                   steps after clearing).  Failure (-1): severity 2 or 3,
                   passage abandoned (|XTE| > abandon_xte for abandon_steps),
                   or leaving the map.  No destination point, no timeout.
  Both: a generous safety cap (max_steps, default 3 h) ends runaway episodes
  with flag 0 and termination 'unresolved'; those are reported separately
  and never counted as success or failure.

Observation (19 values, all O(1), see OBS_NAMES): the elapsed-time fraction
and the destination features of v2 are replaced by the closing rate, the
encounter phase, and the course error against the local route course.

Target motion (target_ship_v3.py): the target HOLDS course and speed on
the approach.  Scenario-family routes are straight (route_shape='straight',
the default); the data route is straightened to its chord when
route_shape='straight', kept when 'data', and decided by `straighten_tol`
when 'auto' (50 m over the whole route since 2026-09-24, mending plan part 1;
with data_prep arrays pass trace_dt 20 and smooth_sigma 0, the target speed is
then the smoother's median speed of the window).  The target alters course once
per emergency (COLREG starboard
turn) and observes a cooldown before it may alter again.

Everything else - vessel dynamics, danger criteria, automation levels,
scenario families, grading - is imported unchanged from Ship_envre_v2.
step(a) -> obs, reward, done, success_flag (1 / 0 / -1).
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(os.path.dirname(HERE), 'Ship_envre_v2')
if V2 not in sys.path:
    sys.path.insert(0, V2)

from ship_dynamics import ShipDynamics                                     # noqa: E402
from encounter import EncounterParams, assess, wrap_pi, SEVERITY_NAMES       # noqa: E402
from target_ship import TargetShip, smooth_path, route_from_trace           # noqa: E402
from scenarios import SCENARIOS, make_scenario, own_speed_range, colreg_role, extend_route  # noqa: E402
from grading import grade_event                                              # noqa: E402
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from target_ship_v3 import steady_target_from_level, straighten_route, route_deviation   # noqa: E402
from ship_env_v2 import (load_trace_set, load_trace_speeds, lonlat_to_xy, cumulative_length, project_on_polyline,  # noqa: E402
                         make_box, make_discrete, OwnShipInit, TargetShipSpec, _discrete_from)

NM = 1852.0
PHASE_NAMES = {0: 'approach', 1: 'action', 2: 'passing', 3: 'clear'}

OBS_NAMES = ['range', 'cos_brg', 'sin_brg', 'surge', 'sway', 'yaw_rate', 'rel_vx_b', 'rel_vy_b',
             'cos_dpsi', 'sin_dpsi', 'closing_rate', 'dcpa', 'tcpa', 'severity',
             'phase', 'cos_course_err', 'sin_course_err', 'xte', 'prev_rudder']


def route_course_at(route, i_seg):
    seg = route[min(i_seg + 1, len(route) - 1)] - route[min(i_seg, len(route) - 2)]
    return float(np.arctan2(seg[1], seg[0]))


def make_obs_v3(state, own_v, other_pos, other_v, other_psi, enc, closing_rate, phase, course_err, prev_rudder,
                params, d_scale, v_scale, r_scale):
    x, y, psi, u, v, r = state
    rel = other_v - own_v
    c, s = np.cos(psi), np.sin(psi)
    rel_b = np.array([c * rel[0] + s * rel[1], -s * rel[0] + c * rel[1]])
    dpsi = wrap_pi(other_psi - psi)
    tcpa = enc['tcpa'] if np.isfinite(enc['tcpa']) else 5 * params.t_safe
    bearing = wrap_pi(np.arctan2(other_pos[1] - y, other_pos[0] - x) - psi)
    obs = np.array([
        enc['dist'] / d_scale, np.cos(bearing), np.sin(bearing),
        u / v_scale, v / v_scale, r / r_scale,
        rel_b[0] / v_scale, rel_b[1] / v_scale,
        np.cos(dpsi), np.sin(dpsi), np.clip(closing_rate / v_scale, -2.0, 2.0),
        min(enc['dcpa'] / params.d_safe, 5.0), np.clip(tcpa / params.t_safe, -1.0, 5.0),
        enc['severity'] / 3.0, phase / 3.0,
        np.cos(course_err), np.sin(course_err),
        np.clip(enc['xte'] / 500.0, -5.0, 5.0), prev_rudder,
    ], dtype=np.float32)
    return np.clip(obs, -5, 5)


class EncounterRules:
    """Parameters of the lifecycle definition."""
    def __init__(self, r_clear=2 * NM, n_clear=30, xte_tol=0.1 * NM, course_tol_deg=10.0, recover_steps=18,
                 abandon_xte=1.0 * NM, abandon_steps=60, route_extension=15 * NM):
        self.r_clear = float(r_clear)
        self.n_clear = int(n_clear)
        self.xte_tol = float(xte_tol)
        self.course_tol = np.radians(course_tol_deg)
        self.recover_steps = int(recover_steps)
        self.abandon_xte = float(abandon_xte)
        self.abandon_steps = int(abandon_steps)
        self.route_extension = float(route_extension)

    def as_dict(self):
        return dict(r_clear=self.r_clear, n_clear=self.n_clear, xte_tol=self.xte_tol,
                    course_tol_deg=float(np.degrees(self.course_tol)), recover_steps=self.recover_steps,
                    abandon_xte=self.abandon_xte, abandon_steps=self.abandon_steps,
                    route_extension=self.route_extension)


# ---------------------------------------------------- policy-driven ship
class PolicyShipV3:
    """Other ship on the full dynamics steered by a policy trained on the v3
    observation (same layout as the own ship, roles swapped)."""

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
        self.prev_d = None
        self.evading = False
        self.n_evasions = 0
        self.route_done = False
        self.phase = 0
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

    def step(self, dt, p_other=None, v_other=None, psi_other=None):
        env = self.env
        enc = assess(self.pos, self.vel, self.psi, p_other, v_other, psi_other, env.enc_params)
        s, xte, self.i_route = project_on_polyline(self.pos, self.route, self.cum_s, self.i_route)
        enc['xte'] = xte
        closing = 0.0 if self.prev_d is None else (self.prev_d - enc['dist']) / dt
        self.prev_d = enc['dist']
        if enc['severity'] >= 1:
            self.phase = max(self.phase, 1)
        if enc['tcpa'] < 0 and self.phase >= 1:
            self.phase = max(self.phase, 2)
        course_err = wrap_pi(route_course_at(self.route, self.i_route) - self.psi)
        obs = make_obs_v3(self.dyn.state, self.vel, np.asarray(p_other), np.asarray(v_other), psi_other, enc,
                          closing, self.phase, course_err, self.prev_action[0], env.enc_params,
                          env.d_scale, env.v_scale, env.r_scale)
        rd, th = env._decode_action(self.policy(obs))
        was = self.evading
        self.evading = enc['severity'] >= 1
        if self.evading and not was:
            self.n_evasions += 1
        self.dyn.step(rd, th)
        self.prev_action = np.array([rd, th])
        return self.state()


# ------------------------------------------------------------------ env
class ShipEncounterEnv:
    ACTION_RUDDER = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    ACTION_THROTTLE = np.array([0.3, 0.65, 1.0])
    OBS_NAMES = OBS_NAMES

    def __init__(self, guideline, mergeline, task='attack', own_init=None, target_spec=None, encounter=None,
                 rules=None, scenario='data', cycle_scenarios=False, other_policy=None,
                 route_shape='straight', straighten_tol=50.0, evasion_cooldown=30,
                 decision_interval=10.0, max_steps=1080, border_margin=3000.0,
                 action_mode='discrete', reward_type='shaped',
                 success_severity=2, hold_steps=2,
                 d_scale=5000.0, v_scale=15.0, r_scale=0.02,
                 R_success=50.0, R_fail=20.0, R_danger=30.0, time_cost=0.02,
                 rudder_change_cost=0.05, save_dir=None, seed=None):
        assert task in ('attack', 'navigate')
        self.task = task
        self.own_init = own_init or OwnShipInit()
        self.target_spec = target_spec or TargetShipSpec()
        self.dyn = ShipDynamics(decision_interval=decision_interval)
        self.enc_params = encounter or EncounterParams(L=self.dyn.L)
        self.rules = rules or EncounterRules()
        self.scenarios = [scenario] if isinstance(scenario, str) else list(scenario)
        if 'mix' in self.scenarios:
            self.scenarios = list(SCENARIOS)
        for sname in self.scenarios:
            assert sname in SCENARIOS, 'unknown scenario %s' % sname
        self.cycle_scenarios = bool(cycle_scenarios)
        self.other_policy = other_policy
        assert route_shape in ('straight', 'data', 'auto')
        self.route_shape = route_shape
        self.straighten_tol = float(straighten_tol)
        self.evasion_cooldown = int(evasion_cooldown)
        self.route_straightened = False
        self.dt = float(decision_interval)
        self.max_steps = int(max_steps)
        self.border_margin = float(border_margin)
        self.action_mode = action_mode
        self.reward_type = reward_type
        self.success_severity = int(success_severity)
        self.hold_steps = int(hold_steps)
        self.d_scale, self.v_scale, self.r_scale = d_scale, v_scale, r_scale
        self.R_success, self.R_fail, self.R_danger = R_success, R_fail, R_danger
        self.time_cost, self.rudder_change_cost = time_cost, rudder_change_cost
        self.save_dir = save_dir
        self.rng = np.random.default_rng(seed)
        self.viewer = None
        self.recorder = None

        g = load_trace_set(guideline)
        m = load_trace_set(mergeline)
        self.merge_speeds = load_trace_speeds(mergeline, m.shape[0])   # data_prep sets only, else None
        self.lon0 = float(np.mean(g[:, 0, 0]))
        self.lat0 = float(np.mean(g[:, 0, 1]))
        self.guidelines = lonlat_to_xy(g, self.lon0, self.lat0)
        self.mergelines = lonlat_to_xy(m, self.lon0, self.lat0)
        self.n_guide, self.n_merge = self.guidelines.shape[0], self.mergelines.shape[0]
        allpts = np.concatenate([self.guidelines.reshape(-1, 2), self.mergelines.reshape(-1, 2)])
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
        self.phase = 0
        self.clear_count = 0
        self.recover_count = 0
        self.abandon_count = 0
        self.t_warning = None
        self.t_cpa = None
        self.t_clear = None
        self.t_on_track = None
        self.max_severity = 0
        self.sev_steps = {1: 0, 2: 0, 3: 0}
        self.first_encounter = None
        self.psi_at_warning = None
        self.course_change_after_warning = None
        self.own_path, self.target_path, self.target_psi = [], [], []
        self.ownship_action, self.rewards, self.dists = [], [], []
        self.enc_hist, self.evade_hist, self.phase_hist = [], [], []
        self.min_distance = np.inf
        self.min_dcpa = np.inf
        self.closing_at_min = 0.0
        self.max_cri = 0.0
        self.max_xte_after_passing = 0.0
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
        enc['course_err'] = float(wrap_pi(route_course_at(self.guide_route, self.i_guide) - psi))
        enc['bearing'] = float(wrap_pi(np.arctan2(self.target.pos[1] - y, self.target.pos[0] - x) - psi))
        enc['own_speed'] = float(np.linalg.norm(own_v))
        enc['target_speed'] = float(self.target.speed)
        enc['target_evading'] = bool(self.target.evading)
        enc['dest_range'] = 0.0
        return enc

    def _observe(self, enc, closing):
        return make_obs_v3(self.dyn.state, self.dyn.global_velocity(), self.target.pos, self.target.vel,
                           self.target.psi, enc, closing, self.phase, enc['course_err'], self.prev_action[0],
                           self.enc_params, self.d_scale, self.v_scale, self.r_scale)

    def _build_target(self, route, speed):
        ts = self.target_spec
        if self.other_policy is not None:
            return PolicyShipV3(self, route, speed, self.other_policy)
        kw = dict(trace_dt=ts.trace_dt, yaw_rate_max_deg=ts.yaw_rate_max_deg, lookahead=ts.lookahead,
                  clear_steps=ts.clear_steps, cooldown_steps=self.evasion_cooldown)
        kw.update(ts.overrides)
        return steady_target_from_level(ts.automation, route, speed, **kw)

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

        # own ship: open-ended route (generated guideline extended along its final course)
        self.gi = int(self.rng.integers(self.n_guide))
        self.mi = int(self.rng.integers(self.n_merge))
        self.guideline = self.guidelines[self.gi]
        self.guide_route = extend_route(smooth_path(self.guideline, ts.smooth_sigma), self.rules.route_extension)
        self.guide_s = cumulative_length(self.guide_route)
        self.i_guide = 0
        self.dest = self.guide_route[-1].copy()          # drawing compatibility only, no destination logic
        self.dest_radius = 0.0
        k = self.own_init.start_index
        p0 = self.guide_route[k].copy()
        if self.own_init.position_jitter_m > 0:
            p0 += self.rng.normal(0, self.own_init.position_jitter_m, 2)
        seg = self.guide_route[min(k + 1, len(self.guide_route) - 1)] - self.guide_route[k]
        psi0 = np.arctan2(seg[1], seg[0]) + self.rng.uniform(-self.own_init.heading_noise, self.own_init.heading_noise)
        speed0 = float(self.rng.uniform(*own_speed_range(self.scenario, self.own_init.speed_range)))
        self.init_speed, self.init_heading = speed0, psi0
        self.dyn.reset(p0[0], p0[1], psi0, speed0)

        # target ship: open-ended route + speed
        merge = self.mergelines[self.mi]
        if ts.speed_scale_range is not None:
            self._speed_scale = float(self.rng.uniform(*ts.speed_scale_range))
        else:
            self._speed_scale = ts.speed_scale
        self.route_straightened = False
        if self.scenario == 'data':
            self.target_trace = merge[::-1].copy() if ts.reverse else merge.copy()
            route, speed = route_from_trace(self.target_trace, ts.trace_dt, ts.smooth_sigma, self._speed_scale)
            speed_source = 'route length over trace duration'
            if self.merge_speeds is not None and np.isfinite(self.merge_speeds[self.mi]):
                # data_prep set: the median speed of the smoother on this window (mending plan part 1)
                speed = float(self.merge_speeds[self.mi]) * self._speed_scale
                speed_source = 'data_prep windows_speed_course median'
            dev = route_deviation(route)                      # whole data route, before the extension
            if self.route_shape == 'straight':
                route, self.route_straightened = straighten_route(route)
            elif self.route_shape == 'auto':
                route, self.route_straightened = straighten_route(route, tol=self.straighten_tol)
            route = extend_route(route, self.rules.route_extension)
            self.scenario_meta = dict(scenario='data', target_speed=speed, speed_source=speed_source,
                                      route_deviation_m=dev, route_straightened=int(self.route_straightened))
        else:
            shape = None if self.route_shape == 'straight' else smooth_path(merge, ts.smooth_sigma)
            sc = make_scenario(self.scenario, p0, psi0, speed0, self.rng, shape_trace=shape)
            route = sc['route']
            if self.route_shape == 'auto':
                route, self.route_straightened = straighten_route(route, tol=self.straighten_tol)
            else:
                self.route_straightened = self.route_shape == 'straight'
            dev = route_deviation(route)                      # the scenario route, before the extension
            route, speed = extend_route(route, self.rules.route_extension), sc['speed'] * self._speed_scale
            self.target_trace = route
            self.scenario_meta = dict(sc['meta'], route_deviation_m=dev,
                                      route_straightened=int(self.route_straightened))
        self.target = self._build_target(route, speed)
        self.target.reset()
        self.route_speed = speed
        # map: the region both ships can reach within the safety cap when sailing straight, so that
        # leaving the map only ever means wandering off sideways, never a long but regular passage
        horizon = self.max_steps * self.dt
        own_reach = p0 + speed0 * horizon * np.array([np.cos(psi0), np.sin(psi0)])
        tgt_reach = self.target.pos + speed * horizon * np.array([np.cos(self.target.psi), np.sin(self.target.psi)])
        self._set_borders(np.stack([p0, own_reach, self.target.pos, tgt_reach]))

        enc = self._assess()
        self._record(enc)
        self.prev_d, self.prev_dcpa, self.prev_s = enc['dist'], self._dcpa_for_shaping(enc), enc['progress']
        if self.viewer is not None:
            self.viewer.end_episode()
        if self.recorder is not None:
            self.recorder.new_iter(self.dyn.state.copy(), self._observe(enc, 0.0), np.zeros(2), np.array([0.0]),
                                   self.target.state(), self.phase, self.guide_route)
        return self._observe(enc, 0.0)

    def _dcpa_for_shaping(self, enc):
        return enc['dcpa'] if np.isfinite(enc['tcpa']) and enc['tcpa'] >= 0 else enc['dist']

    def _record(self, enc):
        self.own_path.append(self.dyn.state[:2].copy())
        self.target_path.append(self.target.pos.copy())
        self.target_psi.append(self.target.psi)
        self.dists.append(enc['dist'])
        self.enc_hist.append(enc)
        self.evade_hist.append(bool(self.target.evading))
        self.phase_hist.append(self.phase)
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
                self.t_warning = self.step_count * self.dt
                self.psi_at_warning = float(self.dyn.state[2])
                if self.colreg_role == 'unknown':
                    self.colreg_role = {'head_on': 'both_give_way', 'crossing_starboard': 'give_way',
                                        'crossing_port': 'stand_on', 'overtaking': 'give_way',
                                        'overtaken': 'stand_on'}.get(enc['encounter'], 'unknown')
        if (self.first_encounter is not None and self.course_change_after_warning is None
                and self.step_count - self.first_encounter[0] >= 30):
            self.course_change_after_warning = float(np.degrees(wrap_pi(self.dyn.state[2] - self.psi_at_warning)))
        if self.phase >= 2:
            self.max_xte_after_passing = max(self.max_xte_after_passing, abs(enc['xte']))
        self.max_severity = max(self.max_severity, sev)

    def _update_phase(self, enc, d):
        """Lifecycle bookkeeping; returns True when the encounter is clear."""
        sev = enc['severity']
        cpa_behind = (enc['tcpa'] < 0) or not np.isfinite(enc['tcpa'])
        if sev >= 1 and self.phase < 1:
            self.phase = 1
        if cpa_behind and self.phase >= 1 and self.phase < 2:
            self.phase = 2
            self.t_cpa = self.step_count * self.dt
        opening = d > self.prev_d
        if sev == 0 and d > self.rules.r_clear and opening and (cpa_behind or self.phase >= 1 or enc['dcpa'] >= self.enc_params.d_safe):
            self.clear_count += 1
        else:
            self.clear_count = 0
        if self.clear_count >= self.rules.n_clear and self.phase < 3:
            self.phase = 3
            self.t_clear = self.step_count * self.dt
        return self.phase == 3

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
        d = enc['dist']
        clear = self._update_phase(enc, d)
        self._record(enc)
        sev = enc['severity']
        own_p = self.dyn.state[:2]
        dcpa_s = self._dcpa_for_shaping(enc)
        closing = (self.prev_d - d) / self.dt

        # track-keeping bookkeeping (navigation)
        on_track = abs(enc['xte']) < self.rules.xte_tol and abs(enc['course_err']) < self.rules.course_tol
        self.recover_count = self.recover_count + 1 if (clear and on_track) else 0
        self.abandon_count = self.abandon_count + 1 if abs(enc['xte']) > self.rules.abandon_xte else 0
        if self.recover_count >= self.rules.recover_steps and self.t_on_track is None:
            self.t_on_track = self.step_count * self.dt

        # ---- termination -------------------------------------------------
        success_flag = 0
        if self.task == 'attack':
            self.hold = self.hold + 1 if sev >= self.success_severity else 0
            if sev == 3 or self.hold >= self.hold_steps:
                self.done, success_flag, self.termination = True, 1, SEVERITY_NAMES[sev]
                self.capture_step = self.step_count
            elif not self._in_borders(own_p):
                self.done, success_flag, self.termination = True, -1, 'own_out_of_map'
            elif clear:
                self.done, success_flag, self.termination = True, -1, 'clear'
            elif self.step_count >= self.max_steps:
                self.done, success_flag, self.termination = True, 0, 'unresolved'
        else:
            if sev >= 2:
                self.done, success_flag, self.termination = True, -1, SEVERITY_NAMES[sev]
            elif self.abandon_count >= self.rules.abandon_steps:
                self.done, success_flag, self.termination = True, -1, 'passage_abandoned'
            elif not self._in_borders(own_p):
                self.done, success_flag, self.termination = True, -1, 'own_out_of_map'
            elif clear and self.recover_count >= self.rules.recover_steps:
                self.done, success_flag, self.termination = True, 1, 'resolved_on_track'
                self.capture_step = self.step_count
            elif self.step_count >= self.max_steps:
                self.done, success_flag, self.termination = True, 0, 'unresolved'

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
            progress = float(np.clip((enc['progress'] - self.prev_s) / 100.0, -1.0, 1.0))
            rew = (0.3 * progress - 0.5 * min(abs(enc['xte']), 1000.0) / 1000.0
                   - 0.5 * (1.0 - np.cos(enc['course_err'])) - self.time_cost - manoeuvre_cost
                   - (1.0 if sev == 1 else 0.0) + (0.2 if (clear and on_track) else 0.0))
        if self.termination in ('cpa_warning', 'domain_violation', 'collision'):
            rew += self.R_success if self.task == 'attack' else -(self.R_danger + (20.0 if sev == 3 else 0.0))
        elif self.termination == 'resolved_on_track':
            rew += self.R_success
        elif self.termination in ('own_out_of_map', 'clear', 'passage_abandoned'):
            rew -= self.R_fail
        self.prev_d, self.prev_dcpa, self.prev_s = d, dcpa_s, enc['progress']
        self.rewards.append(float(rew))
        self.success_flag = success_flag
        self.last_info = dict(enc, termination=self.termination, phase=self.phase)
        obs = self._observe(enc, closing)
        if self.recorder is not None:
            self.recorder.new_transition(self.dyn.state.copy(), obs, self.prev_action.copy(), np.array([rew]),
                                         self.target.state(), self.phase, self.guide_route)
        return obs, float(rew), self.done, success_flag

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
        t_end = self.step_count * self.dt
        ev = dict(
            episode=self.episode_id, task=self.task, scenario=self.scenario,
            automation=getattr(self.target, 'automation', 'custom'), colreg_role=self.colreg_role,
            success=int(self.success_flag == 1), unresolved=int(self.termination == 'unresolved'),
            termination=self.termination, final_phase=PHASE_NAMES[self.phase],
            steps=self.step_count, dt=self.dt,
            capture_step=self.capture_step if self.capture_step is not None else -1,
            destination_step=self.destination_step,
            t_warning=self.t_warning if self.t_warning is not None else -1,
            t_cpa=self.t_cpa if self.t_cpa is not None else -1,
            t_clear=self.t_clear if self.t_clear is not None else -1,
            t_on_track=self.t_on_track if self.t_on_track is not None else -1,
            resolution_time=(self.t_clear - (self.t_warning if self.t_warning is not None else 0.0))
            if self.t_clear is not None else -1,
            recovery_time=(self.t_on_track - self.t_clear) if (self.t_on_track is not None and self.t_clear is not None) else -1,
            max_severity=self.max_severity, steps_sev1=self.sev_steps[1], steps_sev2=self.sev_steps[2],
            steps_sev3=self.sev_steps[3],
            min_distance=float(self.min_distance), min_dcpa=float(self.min_dcpa if np.isfinite(self.min_dcpa) else -1),
            closing_speed_at_min=float(self.closing_at_min), max_cri=float(self.max_cri),
            first_encounter_step=self.first_encounter[0] if self.first_encounter else -1,
            first_encounter_type=self.first_encounter[1] if self.first_encounter else 'none',
            course_change_after_warning_deg=self.course_change_after_warning,
            target_evasions=int(self.target.n_evasions), target_speed=float(self.route_speed),
            final_distance=float(self.dists[-1]), own_path_len=path_len,
            mean_speed=path_len / max(t_end, 1e-9),
            rudder_mean_abs=float(np.mean(np.abs(act[:, 0]))) if len(act) else 0.0,
            rudder_change_mean=float(np.mean(d_act[:, 0])), throttle_change_mean=float(np.mean(d_act[:, 1])),
            xte_mean=float(np.mean(xte)), xte_final=float(abs(self.enc_hist[-1]['xte'])),
            max_xte_after_passing=float(self.max_xte_after_passing),
            progress=float(self.enc_hist[-1]['progress']), route_length=float(self.guide_s[-1]),
            dest_range_final=0.0, total_reward=float(np.sum(self.rewards)),
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
        ax.plot(self.guide_route[:, 0], self.guide_route[:, 1], color='0.6', lw=1, ls='--', label='own route')
        ax.plot(self.target.route[:, 0], self.target.route[:, 1], color='salmon', lw=1, ls='--', label='target route')
        ax.plot(own[:, 0], own[:, 1], 'b-', lw=2, label='own ship')
        ax.plot(tgt[:, 0], tgt[:, 1], 'r-', lw=2, label='target ship')
        ph = np.asarray(self.phase_hist)
        for p_, col in [(1, 'orange'), (2, 'purple'), (3, 'green')]:
            m = ph == p_
            if m.any():
                ax.plot(own[m, 0], own[m, 1], '.', color=col, ms=3, label='phase %s' % PHASE_NAMES[p_])
        ax.plot(own[0, 0], own[0, 1], 'bo'); ax.plot(tgt[0, 0], tgt[0, 1], 'ro')
        ax.plot(own[-1, 0], own[-1, 1], 'bs'); ax.plot(tgt[-1, 0], tgt[-1, 1], 'rs')
        ax.add_patch(Ellipse(tgt[-1], 2 * self.enc_params.domain_a, 2 * self.enc_params.domain_b,
                             angle=np.degrees(self.target_psi[-1]), fc='none', ec='r', ls=':', lw=1))
        ax.add_patch(Circle(tgt[-1], self.enc_params.d_safe, fc='none', ec='orange', ls=':', lw=0.7))
        lo = np.minimum(own.min(axis=0), tgt.min(axis=0)) - 1500
        hi = np.maximum(own.max(axis=0), tgt.max(axis=0)) + 1500
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
        ax.set_title(title or '%s / %s ep %d: %s (%s), %d steps, max sev %d, min DCPA %.0f m' % (
            self.task, self.scenario, self.episode_id, self.termination, PHASE_NAMES[self.phase], self.step_count,
            self.max_severity, self.min_dcpa if np.isfinite(self.min_dcpa) else -1))
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

    # ----------------------------------------------------- ShipAI elements
    def render(self, mode='human'):
        """Live turtle viewer adapted from SimpleShipAI/viewer.py (viewer_v3.py)."""
        if self.viewer is None:
            from viewer_v3 import Viewer
            self.viewer = Viewer(self.borders)
            self.viewer.plot_route(self.guide_route, 'gray')
            self.viewer.plot_route(self.target.route, 'salmon')
        self.viewer.plot_positions(self.dyn.state, self.target.state(), 30 * self.prev_action[0],
                                   self.enc_hist[-1]['severity'], self.phase)

    def set_recorder(self, recorder):
        """Attach a ShipExperiment-style recorder (ship_data_v3.py)."""
        self.recorder = recorder

    def close(self):
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None


# ------------------------------------------------------- scripted policies
def pursuit_policy(env, obs):
    bearing = np.arctan2(obs[2], obs[1])
    rd = float(np.clip(bearing / np.radians(20), -1, 1))
    if env.action_mode == 'discrete':
        return _discrete_from(env, rd, 1.0)
    return np.array([rd, 1.0])


def guideline_policy(env, obs):
    """Navigation baseline: pure pursuit on the route, mid throttle, starboard turn while the CPA warning is active."""
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
    from make_synthetic_data import make_synthetic_traces
    g, m = make_synthetic_traces(n=50, seed=0)
    for task, policy in [('attack', pursuit_policy), ('navigate', guideline_policy)]:
        env = ShipEncounterEnv(g, m, task=task, scenario='mix', cycle_scenarios=True,
                               target_spec=TargetShipSpec(automation='assisted'), seed=1)
        res = []
        for ep in range(len(SCENARIOS)):
            obs = env.reset(); done = False
            while not done:
                obs, r, done, flag = env.step(policy(env, obs))
            e = env.evaluation()
            res.append('%s:%s/%d/%s/res %.0fs/rec %.0fs/g%d' % (
                e['scenario'][:8], e['termination'][:10], e['steps'], e['final_phase'], e['resolution_time'],
                e['recovery_time'], e['event_grade']))
        print(task, res)
        env.show_scenes(os.path.join(HERE, 'scene_smoke_%s.png' % task))
