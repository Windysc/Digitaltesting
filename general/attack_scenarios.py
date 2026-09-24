"""
attack_scenarios.py -- multiple attack scenarios under the Ship_envre_v3 scenario
definition, with the Ship_envre_v2 collision standard (encounter_standard.py).

Definition (the "encounter lifecycle" of the earlier Ship_envre_v3 build): an episode is ONE
encounter.  Neither ship has a destination and there is no time limit;
the encounter runs approach -> action -> passing -> clear.
  success (+1)  severity >= success_severity (2 = domain violation) held for
                hold_steps (2) consecutive decisions, or a collision (3)
  failure (-1)  the encounter clears without the event ('clear'), or the
                attacker leaves the map ('off_map')
  neither (0)   'unresolved' at the safety cap (6 x the longest meeting time),
                reported separately
Severity per decision = the worst value over its 1 s sub-steps, so a
collision cannot be stepped over.

Target: NOT an agent, sails a FIXED TRACK (route playback) at the cruise speed;
routes may be shaped by generated traces (--trace).  A trace becomes a route
through ais_prep.py (mending plan part 1, step 4, 2026-09-24): C2
spline table every 5 m, scaled by the standard's scale like every other length,
checked against the turn rate (traces that cannot meet it within 50 m are left
out; the count is in env.trace_info), placed along its own start tangent and
played back from the table.

Attacker control (the six parameters of ownship(lat, long, sp, cog, a, rot)):
  control='full' (default)  acceleration x turn-rate levels, each parameter limited:
      long_range, lat_range   position limits [m] (default: the encounter map); leaving = off_map
      v_range                 speed limits [m/s] (default 3-12, the v2 tanker's steady speeds)
      a_range                 acceleration limits [m/s^2] (default -0.1..+0.1, the reference scripts)
      rot limit               turn rate +- turn_rate_deg [deg/s] (default: turning radius 5 L at cruise)
      cog_limit               max course change from the start course [deg] (default 180 = free)
      acc_levels, rot_levels  odd numbers of discrete levels (default 3 x 3 = 9 actions, zero included)
  control='turn'            the 2026-09-15 preset: cruise speed fixed, 3 turn actions
                            (runs/enc_base checkpoints)

Attack scenarios (geometry of v2 scenarios.py: course difference, meeting time,
lateral offset of the meeting point; side = where the target passes):

  head_on_port              Rule 14  reciprocal courses, target passes down the attacker's port side
  head_on_starboard         Rule 14  reciprocal courses, target passes down the starboard side
  crossing_starboard_bow    Rule 15  target from starboard crosses ahead of the attacker (attacker gives way)
  crossing_starboard_stern  Rule 15  target from starboard passes astern of the attacker
  crossing_port_bow         Rule 15  target from port crosses ahead of the attacker (attacker stands on)
  crossing_port_stern       Rule 15  target from port passes astern of the attacker
  parallel_port             lanes    same course, equal speed, target abeam on port, slightly astern
  parallel_starboard        lanes    same course, equal speed, target abeam on starboard, slightly astern
Groups: head_on, crossing_starboard, crossing_port, crossing, parallel, mix (all 8).

Initial DCPA band (with both ships holding course), in units of the standard:
  collision  [0, collision distance)      holding course already collides
  close      [domain_b, d_safe)           CPA warning, no domain infringement
  passing    [d_safe, 2 d_safe)           safe passage: the attacker must create the danger (default)
  wide       [2 d_safe, 4 d_safe)
  v2         meeting-point offset U(0, 0.3 nm x scale) as in Ship_envre_v2 (mostly collision/close)
The meeting-point offset is solved so the initial DCPA is exactly the drawn value.
"""
import copy
import math
import os

import numpy as np

import env_moving_obj as E
import encounter_standard as S
from env_moving_attack import MassTestingEnv as _AttackEnv
from scenario_targets import (TargetShip, load_trace_shapes, prepare_trace_routes, place_trace_route,
                              route_polyline, route_meta)

ATTACK_SCENARIOS = {
    'head_on_port':             dict(family='head_on', side=+1, rule='Rule 14',
                                     text='reciprocal courses, target passes down the attacker port side'),
    'head_on_starboard':        dict(family='head_on', side=-1, rule='Rule 14',
                                     text='reciprocal courses, target passes down the attacker starboard side'),
    'crossing_starboard_bow':   dict(family='crossing_starboard', side=+1, rule='Rule 15, attacker gives way',
                                     text='target from starboard crosses ahead of the attacker'),
    'crossing_starboard_stern': dict(family='crossing_starboard', side=-1, rule='Rule 15, attacker gives way',
                                     text='target from starboard passes astern of the attacker'),
    'crossing_port_bow':        dict(family='crossing_port', side=-1, rule='Rule 15, attacker stands on',
                                     text='target from port crosses ahead of the attacker'),
    'crossing_port_stern':      dict(family='crossing_port', side=+1, rule='Rule 15, attacker stands on',
                                     text='target from port passes astern of the attacker'),
    'parallel_port':            dict(family='parallel', side=+1, rule='parallel lanes, equal speed',
                                     text='same course, target abeam on the port side, slightly astern'),
    'parallel_starboard':       dict(family='parallel', side=-1, rule='parallel lanes, equal speed',
                                     text='same course, target abeam on the starboard side, slightly astern'),
}
SCENARIO_NAMES = list(ATTACK_SCENARIOS)
GROUPS = {
    'mix': SCENARIO_NAMES,
    'head_on': ['head_on_port', 'head_on_starboard'],
    'crossing_starboard': ['crossing_starboard_bow', 'crossing_starboard_stern'],
    'crossing_port': ['crossing_port_bow', 'crossing_port_stern'],
    'crossing': ['crossing_starboard_bow', 'crossing_starboard_stern', 'crossing_port_bow', 'crossing_port_stern'],
    'parallel': ['parallel_port', 'parallel_starboard'],
}
FAMILY_COURSE = {'head_on': (174.0, 186.0), 'crossing_starboard': (60.0, 120.0), 'crossing_port': (-120.0, -60.0),
                 'parallel': (0.0, 0.0)}
BANDS = ('collision', 'close', 'passing', 'wide', 'v2')
PARALLEL_LAG = (0.5, 2.0)          # target starts this many domain_a astern of the attacker (parallel lanes)


def resolve_scenarios(spec):
    names = []
    for part in str(spec).split(','):
        part = part.strip()
        if part in GROUPS:
            names += GROUPS[part]
        elif part in ATTACK_SCENARIOS:
            names.append(part)
        else:
            raise ValueError('scenario %r unknown; choose from %s or groups %s' % (part, SCENARIO_NAMES, sorted(GROUPS)))
    return list(dict.fromkeys(names))


def band_range(band, std):
    return {'collision': (0.0, std.collision_dist), 'close': (std.domain_b, std.d_safe),
            'passing': (std.d_safe, 2.0 * std.d_safe), 'wide': (2.0 * std.d_safe, 4.0 * std.d_safe)}.get(band)


def make_attack_scenario(name, std, speed, rng, band='passing', own_pos=(0.0, 0.0), own_cog_deg=0.0,
                         shape_trace=None, cap_s=None):
    """Target route + metadata; the target holds the route at `speed` (fixed track)."""
    if name not in ATTACK_SCENARIOS:
        raise ValueError('unknown attack scenario %r' % name)
    if band not in BANDS:
        raise ValueError('dcpa band %r unknown; choose %s' % (band, BANDS))
    spec = ATTACK_SCENARIOS[name]
    fam, side = spec['family'], spec['side']
    cap_s = float(cap_s if cap_s else std.cap_s)
    own_pos = np.asarray(own_pos, float)
    own_psi = math.radians(own_cog_deg)
    d_own = np.array([math.cos(own_psi), math.sin(own_psi)])
    n_own = np.array([-d_own[1], d_own[0]])                        # port side
    lo_hi = band_range(band, std)
    dpsi = math.radians(rng.uniform(*FAMILY_COURSE[fam])) if FAMILY_COURSE[fam][1] > FAMILY_COURSE[fam][0] else 0.0
    psi_t = own_psi + dpsi
    d_t = np.array([math.cos(psi_t), math.sin(psi_t)])
    if fam == 'parallel':
        lateral = rng.uniform(*lo_hi) if lo_hi else rng.uniform(std.collision_dist, std.v2_dcpa_offset)
        lag = rng.uniform(*PARALLEL_LAG) * std.domain_a
        start = own_pos + n_own * side * lateral - d_own * lag
        t_meet, offset, target_dcpa = -1.0, side * lateral, math.hypot(lateral, lag)
    else:
        t_meet = float(rng.uniform(*std.t_meet))
        vr = speed * (d_t - d_own)
        vr_hat = vr / max(np.linalg.norm(vr), 1e-9)
        k = max(abs(n_own[0] * vr_hat[1] - n_own[1] * vr_hat[0]), 0.2)   # |DCPA| per metre of offset
        if lo_hi:
            target_dcpa = float(rng.uniform(*lo_hi))
            offset = side * target_dcpa / k
        else:                                                             # v2 sampling of the offset
            offset = side * float(rng.uniform(0.0, std.v2_dcpa_offset))
            target_dcpa = abs(offset) * k
        meet = own_pos + d_own * speed * t_meet + n_own * offset
        start = meet - d_t * speed * t_meet
    run = speed * (cap_s + 120.0)
    table = None
    if shape_trace is not None and len(shape_trace) > 3:
        # route table from prepare_trace_routes (or a raw trace, built without the limit check), scaled by the
        # standard's scale, turned onto psi_t along its own start tangent, extended straight to the cap
        table = place_trace_route(shape_trace, start, psi_t, run, scale=std.scale)
        route = route_polyline(table)
    else:
        route = np.stack([start, start + d_t * run])
    a0 = S.assess(own_pos, d_own * speed, own_psi, start, d_t * speed, psi_t, std)
    brg = math.degrees(a0['bearing'])
    meta = dict(scenario=name, family=fam, side='port' if side > 0 else 'starboard', rule=spec['rule'], band=band,
                course_diff_deg=round(math.degrees(dpsi), 1), t_meet_s=round(t_meet, 1),
                offset_m=round(float(offset), 1), target_dcpa_m=round(float(target_dcpa), 1),
                initial_dcpa_m=round(a0['dcpa'], 1),
                initial_tcpa_s=round(a0['tcpa'], 1) if math.isfinite(a0['tcpa']) else -1.0,
                initial_range_m=round(a0['dist'], 1), initial_bearing_deg=round(brg, 1),
                initial_severity=int(a0['severity']), initial_encounter=a0['encounter'], **route_meta(table))
    return dict(route=route, route_table=table, speed=float(speed), meta=meta)


# ------------------------------------------------------------------ env
OBS_NAMES = ['own_speed', 'cos_cog', 'sin_cog', 'rel_x', 'rel_y', 'range', 'cos_bearing', 'sin_bearing',
             'cos_course_diff', 'sin_course_diff', 'rel_vx', 'rel_vy', 'closing_rate', 'dcpa', 'tcpa',
             'severity', 'phase', 'domain_margin', 'cri', 'hold']
OBS_FULL = ['speed_in_range', 'prev_accel', 'prev_turn', 'area_margin', 'course_change']   # control='full' only
DEFAULT_V_RANGE = (3.0, 12.0)
DEFAULT_A_RANGE = (-0.1, 0.1)


def _levels(lo, hi, n):
    """n (odd) levels from lo to hi that always contain 0."""
    n = max(1, int(n))
    if n % 2 == 0:
        raise ValueError('number of levels must be odd so that 0 is a level, got %d' % n)
    k = (n - 1) // 2
    if k == 0:
        return [0.0]
    return [float(v) for v in np.linspace(lo, 0.0, k + 1)[:-1]] + [0.0] + [float(v) for v in np.linspace(0.0, hi, k + 1)[1:]]


def action_name(a, r, a_lo, a_hi, R):
    if abs(r) < 1e-12:
        rn = 'straight'
    elif abs(abs(r) - R) < 1e-9:
        rn = 'port' if r > 0 else 'starboard'
    else:
        rn = '%s%.2f' % ('port' if r > 0 else 'stbd', abs(r))
    if a_lo == 0.0 and a_hi == 0.0:
        return rn
    if abs(a) < 1e-12:
        an = 'hold'
    elif abs(a - a_hi) < 1e-12:
        an = 'accel'
    elif abs(a - a_lo) < 1e-12:
        an = 'decel'
    else:
        an = 'a%+.3f' % a
    return an + '/' + rn
OUTCOMES = ('running', 'success', 'clear', 'off_map', 'unresolved')


class EncounterAttackEnv(_AttackEnv):
    """Attack task, one encounter per episode, v2 collision standard, v3 lifecycle, fixed-track target."""

    per_episode_targets = True

    def __init__(self, own_ship, scenario='mix', dcpa_band='passing', scale='arena', cruise_speed=6.0,
                 turn_rate_deg=None, trace=None, cycle_scenarios=False, success_severity=None, hold_steps=None,
                 max_encounter_s=None, decision_interval=600, reward_type='final_attack_reward', save_dir='.',
                 seed=0, duration=None, X_LEN=None, Y_LEN=None, attack_range=None, control='full', v_range=None,
                 a_range=None, cog_limit=180.0, long_range=None, lat_range=None, acc_levels=3, rot_levels=3,
                 **std_overrides):
        self.names = resolve_scenarios(scenario)
        self.families = self.names                       # viz / scripts treat these as the scenario list
        self.scenario_spec = scenario
        if dcpa_band not in BANDS:
            raise ValueError('dcpa_band %r unknown; choose %s' % (dcpa_band, BANDS))
        self.band = dcpa_band
        self.cruise = float(cruise_speed)
        self.std = S.CollisionStandard(scale, self.cruise, success_severity=success_severity, hold_steps=hold_steps,
                                       **std_overrides)
        self.turn_rate = float(turn_rate_deg) if turn_rate_deg else round(self.std.turn_rate_deg, 3)
        self.automation = 'fixed'
        self.trace_path = trace or None
        self.trace_shapes = load_trace_shapes(trace) if trace else None
        # routes the target can sail at cruise speed under the turn rate once scaled like every other length
        self.trace_routes, self.trace_info = (
            prepare_trace_routes(self.trace_shapes, self.cruise, self.turn_rate, scale=self.std.scale)
            if self.trace_shapes else (None, None))
        self.cycle = bool(cycle_scenarios)
        self._cycle_i = 0
        self.scen_rng = np.random.RandomState(seed)
        self.current = {}
        dt = int(decision_interval) * E.TICK_S
        cap_s = float(max_encounter_s) if max_encounter_s else self.std.cap_s
        self.cap_decisions = max(1, min(1000, int(math.ceil(cap_s / dt))))   # the scripts' loop runs <= 1000 steps
        if control not in ('full', 'turn'):
            raise ValueError("control must be 'full' or 'turn', got %r" % control)
        self.control = control
        R = self.turn_rate
        own = copy.deepcopy(own_ship)
        own.sp0 = self.cruise
        if control == 'full':
            v_lo, v_hi = (float(v) for v in (v_range or DEFAULT_V_RANGE))
            a_lo, a_hi = (float(v) for v in (a_range or DEFAULT_A_RANGE))
            if not (0.0 <= v_lo <= self.cruise <= v_hi):
                raise ValueError('speed range %s must contain the start (cruise) speed %.2f m/s' % ((v_lo, v_hi), self.cruise))
            if not (a_lo <= 0.0 <= a_hi):
                raise ValueError('acceleration range %s must contain 0' % ((a_lo, a_hi),))
            accs, rots = _levels(a_lo, a_hi, acc_levels), _levels(R, -R, rot_levels)
        else:
            v_lo = v_hi = self.cruise
            a_lo = a_hi = 0.0
            accs, rots = [0.0], [R, 0.0, -R]
        own.v_min, own.v_max = v_lo, v_hi
        own.a_min, own.a_max = a_lo, a_hi
        own.cog_limit = float(cog_limit) if (cog_limit is not None and float(cog_limit) < 180.0) else None
        own.reset()
        self.action_table = [(float(a), float(r)) for a in accs for r in rots]
        self.action_names = [action_name(a, r, a_lo, a_hi, R) for a, r in self.action_table]
        self.long_range = tuple(float(v) for v in long_range) if long_range else None
        self.lat_range = tuple(float(v) for v in lat_range) if lat_range else None
        self.limits = dict(control=control, long_range=self.long_range, lat_range=self.lat_range, v_range=(v_lo, v_hi),
                           a_range=(a_lo, a_hi), rot_range=(-R, R), cog_limit=own.cog_limit or 180.0,
                           acc_levels=len(accs), rot_levels=len(rots))
        self.own_ship_proto = own
        self.dt_decision = dt
        self.target = None
        self._new_scenario(own)
        super().__init__(own, [self.target], [], None, duration=self.cap_decisions * int(decision_interval),
                         decision_interval=decision_interval, reward_type=reward_type,
                         X_LEN=self.X_LEN, Y_LEN=self.Y_LEN, save_dir=save_dir, attack_range=None)
        self.obs_dim = len(OBS_NAMES) + (len(OBS_FULL) if self.control == 'full' else 0)
        self.observation_space = E._make_box(self.obs_dim)
        self.max_decisions = self.cap_decisions
        self._cycle_i = 0
        self.scen_rng = np.random.RandomState(seed)

    # ----------------------------------------------------------- scenarios
    def _new_scenario(self, own_proto=None):
        proto = own_proto if own_proto is not None else self.own_ship_proto
        if self.cycle:
            name = self.names[self._cycle_i % len(self.names)]
            self._cycle_i += 1
        else:
            name = self.names[self.scen_rng.randint(len(self.names))]
        shape = self.trace_routes[self.scen_rng.randint(len(self.trace_routes))] if self.trace_routes else None
        cap_s = self.cap_decisions * self.dt_decision
        sc = make_attack_scenario(name, self.std, self.cruise, self.scen_rng, band=self.band,
                                  own_pos=(proto.long0, proto.lat0), own_cog_deg=proto.cog0, shape_trace=shape,
                                  cap_s=cap_s)
        route = sc['route_table'] if sc.get('route_table') is not None else sc['route']
        self.target = TargetShip(route, self.cruise, L=self.std.L, W=self.std.W, risk_range=self.std.d_safe,
                                 turn_rate_deg=self.turn_rate, automation='fixed', id=name, meta=sc['meta'])
        self.current = dict(sc['meta'])
        self.ts_list, self.ob_list = [self.target], []
        self.objects, self.targets, self.hazards = [self.target], [self.target], []
        # map (v3): the region both ships cover holding course over the cap, plus r_clear
        o = np.array([proto.long0, proto.lat0])
        d_own = np.array([math.cos(math.radians(proto.cog0)), math.sin(math.radians(proto.cog0))])
        route = np.asarray(sc['route'])
        t_pts = [route[0], route[0] + (route[-1] - route[0]) / max(np.linalg.norm(route[-1] - route[0]), 1e-9)
                 * min(self.cruise * cap_s, float(np.linalg.norm(route[-1] - route[0])))]
        pts = np.array([o, o + d_own * self.cruise * cap_s] + t_pts)
        m = self.std.r_clear
        self.map_box = (float(pts[:, 0].min() - m), float(pts[:, 0].max() + m),
                        float(pts[:, 1].min() - m), float(pts[:, 1].max() + m))
        if getattr(self, 'long_range', None):             # explicit position limits of the attacker
            self.map_box = (self.long_range[0], self.long_range[1], self.map_box[2], self.map_box[3])
        if getattr(self, 'lat_range', None):
            self.map_box = (self.map_box[0], self.map_box[1], self.lat_range[0], self.lat_range[1])
        self.X_LEN = self.map_box[1] - self.map_box[0]
        self.Y_LEN = 0.5 * (self.map_box[3] - self.map_box[2])

    def seed(self, seed=None):
        self.rng = np.random.RandomState(seed)
        self.scen_rng = np.random.RandomState(seed)
        return [seed]

    def _in_map(self, x, y):
        b = self.map_box
        return b[0] <= x <= b[1] and b[2] <= y <= b[3]

    # ----------------------------------------------------------- gym api
    def reset(self):
        self._new_scenario()
        own = copy.deepcopy(self.own_ship_proto)
        own.reset()
        self.live_ownship = own
        self.target.reset()
        self.t = 0.0
        self.n_steps = 0
        self.destination_step = 0
        self.outcome = 'running'
        self.termination = 'running'
        self.done = False
        self.ownship_action = []
        self.rewards = []
        self.total_reward = 0.0
        self._episode_started = False
        self.lifecycle = S.Lifecycle(self.std, self.dt_decision)
        enc = self._assess()
        self.enc = dict(enc, phase=0, hold=0)
        self.lifecycle.start(enc['dist'])
        self.d0 = enc['dist']
        self.prev_d = enc['dist']
        self.prev_dcpa = self._dcpa_s(enc)
        self.prev_max = int(enc['severity'])
        self.hold = 0
        self.max_severity = 0
        self.sev_steps = [0, 0, 0, 0]
        self.min_distance = enc['dist']
        self.min_dcpa = enc['dcpa'] if (math.isfinite(enc['tcpa']) and enc['tcpa'] >= 0) else math.inf
        self.max_cri = enc['cri']
        self.closing_at_min = 0.0
        self.first_warning = None
        self.t_domain = None
        self.min_target_dist = self.target.hull_distance(own.x, own.y)
        self.min_hazard_dist = math.inf
        self.risk_steps = 0
        self._closing = 0.0
        self.v_hits = 0
        self.cog_hits = 0
        self.max_hoff = 0.0
        self.prev_cmd = (0.0, 0.0)
        self.enc_hist = [self.enc]
        return self._observe(self.enc)

    def _assess(self):
        own, tgt = self.live_ownship, self.target
        return S.assess((own.x, own.y), own.velocity, math.radians(own.cog), (tgt.x, tgt.y), tgt.velocity,
                        math.radians(tgt.psi), self.std)

    @staticmethod
    def _dcpa_s(enc):
        return enc['dcpa'] if (math.isfinite(enc['tcpa']) and enc['tcpa'] >= 0) else enc['dist']

    def step(self, action):
        if self.done:
            raise RuntimeError('step() called on a finished episode; call reset()')
        if not self._episode_started:
            self.episode += 1
            self._episode_started = True
        action = int(action)
        a_cmd, r_cmd = self.action_table[action]
        own, tgt = self.live_ownship, self.target
        own.command(a_cmd, r_cmd)
        self.ownship_action.append(action)
        sub_max, off_map = 0, False
        for _ in range(self.n_sub):
            own.advance(self.dt_sub)
            self.t += self.dt_sub
            tgt.advance(self.dt_sub, own, self.t)
            e = self._assess()
            sub_max = max(sub_max, e['severity'])
            self.min_target_dist = min(self.min_target_dist, tgt.hull_distance(own.x, own.y))
            if not self._in_map(own.x, own.y):
                off_map = True
                break
            if e['severity'] == 3:
                break
        self.n_steps += 1
        if (a_cmd > 0 and own.sp >= own.v_max - 1e-9) or (a_cmd < 0 and own.sp <= own.v_min + 1e-9):
            self.v_hits += 1                                  # commanded beyond a speed limit
        if own.cog_limit is not None and r_cmd != 0 and abs(own.heading_offset) >= own.cog_limit - 1e-9:
            self.cog_hits += 1                                # commanded beyond the course-change limit
        self.max_hoff = max(self.max_hoff, abs(own.heading_offset))
        self.prev_cmd = (a_cmd, r_cmd)
        enc = self._assess()
        sev = int(sub_max)
        d = enc['dist']
        closing = (self.prev_d - d) / self.dt_decision
        self._closing = closing
        if d < self.min_distance:
            self.min_distance = d
            self.closing_at_min = closing
        if math.isfinite(enc['tcpa']) and enc['tcpa'] >= 0:
            self.min_dcpa = min(self.min_dcpa, enc['dcpa'])
        self.max_cri = max(self.max_cri, enc['cri'])
        if sev > 0:
            self.sev_steps[sev] += 1
            self.risk_steps += 1
            if self.first_warning is None:
                self.first_warning = (self.n_steps, enc['encounter'])
        if sev >= 2 and self.t_domain is None:
            self.t_domain = self.t
        self.max_severity = max(self.max_severity, sev)
        clear = self.lifecycle.update(sev, d, enc['dcpa'], enc['tcpa'])
        self.hold = self.hold + 1 if sev >= self.std.success_severity else 0

        if off_map:                                          # the position limits are hard: checked first
            self.outcome, self.termination, flag = 'off_map', 'off_map', -1
        elif sev == 3 or self.hold >= self.std.hold_steps:
            self.outcome, self.termination, flag = 'success', S.SEVERITY_NAMES[sev], 1
        elif clear:
            self.outcome, self.termination, flag = 'clear', 'clear', -1
        elif self.n_steps >= self.max_decisions:
            self.outcome, self.termination, flag = 'unresolved', 'unresolved', 0
        else:
            flag = 0
        self.done = self.outcome != 'running'
        self.enc = dict(enc, severity=sev, severity_now=enc['severity'], phase=self.lifecycle.phase, hold=self.hold)
        self.enc_hist.append(self.enc)

        reward = self._reward_enc(self.enc)
        self.prev_d = d
        self.prev_dcpa = self._dcpa_s(enc)
        self.prev_max = max(self.prev_max, sev)
        self.rewards.append(reward)
        self.total_reward += reward
        if self.done:
            self.destination_step = self.n_steps
        return self._observe(self.enc), reward, self.done, flag

    def _reward_enc(self, enc):
        terminal = 0.0
        if self.outcome == 'success':
            terminal = E.R_SUCCESS
        elif self.outcome in ('clear', 'off_map'):
            terminal = E.R_FAIL
        elif self.outcome == 'unresolved':
            terminal = E.R_FAIL * min(1.0, self.min_distance / max(self.d0, 1.0))
        if self.reward_family == 'final':
            return terminal
        step_run = self.cruise * self.dt_decision
        r = (0.1 * (self.prev_d - enc['dist']) / step_run
             + 0.1 * float(np.clip((self.prev_dcpa - self._dcpa_s(enc)) / step_run, -2.0, 2.0))
             + 0.02 * math.cos(enc['bearing']) - 0.02)
        sev = enc['severity']
        if sev > self.prev_max:
            r += {1: 1.0, 2: 3.0, 3: 3.0}[sev]
        return r + terminal

    def _observe(self, enc):
        own, tgt = self.live_ownship, self.target
        psi = math.radians(own.cog)
        c, s = math.cos(psi), math.sin(psi)
        ref = 2.0 * self.std.r_clear
        dx, dy = tgt.x - own.x, tgt.y - own.y
        vx, vy = own.velocity
        tvx, tvy = tgt.velocity
        rvx, rvy = tvx - vx, tvy - vy
        dpsi = math.radians(E.wrap_deg(tgt.psi - own.cog))
        tcpa = enc['tcpa']
        tcpa_f = min(tcpa / self.std.t_safe, 3.0) / 3.0 if (math.isfinite(tcpa) and tcpa >= 0) else -0.1
        obs = [own.sp / max(self.cruise, 1e-6), c, s,
               float(np.clip((dx * c + dy * s) / ref, -3, 3)), float(np.clip((-dx * s + dy * c) / ref, -3, 3)),
               min(enc['dist'] / ref, 3.0), math.cos(enc['bearing']), math.sin(enc['bearing']),
               math.cos(dpsi), math.sin(dpsi),
               (rvx * c + rvy * s) / (2 * self.cruise), (-rvx * s + rvy * c) / (2 * self.cruise),
               float(np.clip(self._closing / (2 * self.cruise), -1, 1)),
               min(self._dcpa_s(enc) / self.std.d_safe, 5.0) / 5.0, tcpa_f,
               enc['severity'] / 3.0, self.lifecycle.phase / 3.0, min(enc['margin'], 5.0) / 5.0, enc['cri'],
               self.hold / max(self.std.hold_steps, 1)]
        if self.control == 'full':
            b = self.map_box
            edge = min(own.x - b[0], b[1] - own.x, own.y - b[2], b[3] - own.y)
            span = max(own.v_max - own.v_min, 1e-6)
            obs += [(own.sp - own.v_min) / span, self.prev_cmd[0] / max(abs(own.a_min), abs(own.a_max), 1e-6),
                    self.prev_cmd[1] / max(self.turn_rate, 1e-6), float(np.clip(edge / ref, 0.0, 3.0)) / 3.0,
                    own.heading_offset / (own.cog_limit or 180.0)]
        return np.asarray(obs, dtype=np.float32)

    # ----------------------------------------------------------- reporting
    def target_track(self):
        return np.asarray(self.target.track)

    def evaluation(self):
        own = self.live_ownship
        path = np.asarray(own.path)
        seg = np.diff(path, axis=0) if len(path) > 1 else np.zeros((0, 2))
        r_hist = np.asarray(own.rot) if own.rot else np.zeros(0)
        lc = self.lifecycle
        ev = dict(episode=self.episode, task='attack', reward_type=self.reward_type, steps=self.n_steps, dt=self.dt_decision,
                  sim_time_s=round(self.t, 1), outcome=self.outcome, success=int(self.outcome == 'success'),
                  unresolved=int(self.outcome == 'unresolved'), termination=self.termination,
                  final_phase=S.PHASE_NAMES[lc.phase], max_severity=self.max_severity,
                  steps_sev1=self.sev_steps[1], steps_sev2=self.sev_steps[2], steps_sev3=self.sev_steps[3],
                  min_distance=round(self.min_distance, 1),
                  min_dcpa=round(self.min_dcpa, 1) if math.isfinite(self.min_dcpa) else -1.0,
                  max_cri=round(self.max_cri, 3), closing_speed_at_min=round(self.closing_at_min, 2),
                  t_warning=lc.t_warning if lc.t_warning is not None else -1.0,
                  t_domain=round(self.t_domain, 1) if self.t_domain is not None else -1.0,
                  t_cpa=lc.t_cpa if lc.t_cpa is not None else -1.0,
                  t_clear=lc.t_clear if lc.t_clear is not None else -1.0,
                  first_encounter_type=self.first_warning[1] if self.first_warning else 'none')
        grade, gname, score = S.grade_event(ev, self.std)
        ev.update(event_grade=grade, grade_name=gname, event_score=score)
        ev.update({k: self.current.get(k) for k in ('scenario', 'family', 'side', 'rule', 'band', 'course_diff_deg',
                                                    't_meet_s', 'offset_m', 'target_dcpa_m', 'initial_dcpa_m',
                                                    'initial_tcpa_s', 'initial_range_m', 'initial_bearing_deg',
                                                    'initial_severity', 'initial_encounter', 'data_shaped',
                                                    'route_smooth_m', 'route_max_yaw_deg_s')})
        a_hist = np.asarray(own.a) if own.a else np.zeros(0)
        ev.update(control=self.control, min_speed=round(float(np.min(own.speeds)), 2),
                  max_speed=round(float(np.max(own.speeds)), 2), n_accel_cmds=int((a_hist > 0).sum()),
                  n_decel_cmds=int((a_hist < 0).sum()),
                  mean_abs_accel_cmd=round(float(np.mean(np.abs(a_hist))) if len(a_hist) else 0.0, 4),
                  speed_limit_hits=self.v_hits, course_limit_hits=self.cog_hits,
                  max_course_change_deg=round(self.max_hoff, 1))
        ev.update(final_x=round(own.x, 1), final_y=round(own.y, 1), final_cog_deg=round(own.cog, 1),
                  scale=self.std.scale_name, ship_L=round(self.std.L, 1), d_safe=round(self.std.d_safe, 1),
                  min_target_dist=round(self.min_target_dist, 1), steps_in_risk=self.risk_steps,
                  path_length=round(float(np.hypot(seg[:, 0], seg[:, 1]).sum()) if len(seg) else 0.0, 1),
                  mean_speed=round(float(np.mean(own.speeds)), 2), n_turn_cmds=int((r_hist != 0).sum()),
                  target_evasions=0, destination_step=self.destination_step, total_reward=round(self.total_reward, 3))
        return ev

    def show_scenes(self, save_path=None, title=None):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse

        own = self.live_ownship
        path = np.asarray(own.path)
        tgt = self.target
        track = self.target_track()
        fig, ax = plt.subplots(figsize=(8, 6))
        b = self.map_box
        ax.add_patch(plt.Rectangle((b[0], b[2]), b[1] - b[0], b[3] - b[2], fill=False, ls=':', color='grey'))
        ax.plot(tgt.route[:, 0], tgt.route[:, 1], ':', color='tab:red', alpha=0.4, label='target fixed track')
        ax.plot(track[:, 0], track[:, 1], '-', color='tab:red', lw=1.2, label='target')
        ax.plot(path[:, 0], path[:, 1], '-', color='tab:blue', lw=1.2, label='attacker')
        ax.add_patch(Ellipse((tgt.x, tgt.y), 2 * self.std.domain_a, 2 * self.std.domain_b, angle=tgt.psi,
                             fc='none', ec='tab:red', ls='--'))
        pts = np.concatenate([path, track])
        lo, hi = pts.min(0) - self.std.d_safe, pts.max(0) + self.std.d_safe
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect('equal')
        m = self.current
        ax.set_title(title or '%s / %s | ep %d | %s (%s) in %d decisions | min range %.0f m'
                     % (m.get('scenario'), m.get('band'), self.episode, self.outcome, self.termination, self.n_steps,
                        self.min_distance), fontsize=9)
        ax.legend(loc='upper left', fontsize=8)
        if save_path is None:
            scene_dir = os.path.join(self.save_dir, 'scenes')
            os.makedirs(scene_dir, exist_ok=True)
            self._scene_count += 1
            save_path = os.path.join(scene_dir, 'scene_%s_ep%d_%d.png' % (m.get('scenario'), self.episode, self._scene_count))
        fig.savefig(save_path, dpi=100)
        return fig

    show_path = show_scenes


# ------------------------------------------------------------------ scripted baselines
def hold_action(env, obs=None):
    """No manoeuvre: straight, no acceleration."""
    return steer_action(env, env.live_ownship.cog, 0.0)


def intercept_heading(own_xy, speed, tgt_xy, tgt_v):
    """Collision-course heading at own speed (constant bearing); pure pursuit when no intercept exists."""
    r = np.asarray(tgt_xy, float) - np.asarray(own_xy, float)
    dist = float(np.linalg.norm(r))
    if dist < 1e-6:
        return None
    rh = r / dist
    perp = np.array([-rh[1], rh[0]])
    vt = np.asarray(tgt_v, float)
    vt_par, vt_perp = float(vt @ rh), float(vt @ perp)
    if abs(vt_perp) <= speed:
        sa = vt_perp / speed
        ca = math.sqrt(max(0.0, 1.0 - sa * sa))
        if speed * ca - vt_par > 0.05:
            u = rh * ca + perp * sa
            return math.degrees(math.atan2(u[1], u[0]))
    return math.degrees(math.atan2(rh[1], rh[0]))


def steer_action(env, desired_deg, accel=0.0):
    """Action whose turn brings the course closest to desired_deg after one decision, with the acceleration nearest accel."""
    own = env.live_ownship
    err = E.wrap_deg(desired_deg - own.cog)
    cost = [abs(E.wrap_deg(err - r * env.dt_decision)) + 1e3 * abs(a - accel) for a, r in env.action_table]
    return int(np.argmin(cost))


def intercept_action(env, obs=None):
    """Constant-bearing intercept; with full control at the highest acceleration."""
    own, tgt = env.live_ownship, env.target
    h = intercept_heading((own.x, own.y), own.sp, (tgt.x, tgt.y), tgt.velocity)
    return steer_action(env, own.cog if h is None else h, own.a_max)


def pursuit_action(env, obs=None):
    own, tgt = env.live_ownship, env.target
    return steer_action(env, math.degrees(math.atan2(tgt.y - own.y, tgt.x - own.x)), own.a_max)


BASELINES = {'hold': hold_action, 'intercept': intercept_action, 'pursuit': pursuit_action}

__all__ = ['EncounterAttackEnv', 'ATTACK_SCENARIOS', 'SCENARIO_NAMES', 'GROUPS', 'BANDS', 'OBS_NAMES',
           'make_attack_scenario', 'resolve_scenarios', 'BASELINES', 'intercept_action', 'hold_action',
           'pursuit_action']
