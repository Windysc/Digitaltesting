"""
scenario_targets.py -- multi-scenario attack baseline for the rebuilt MassTestingEnv.

Replaces the ad-hoc straight-line obstacles of main_attack_ppo.py with target
ships generated the way the project's Ship_envre_v2/scenarios.py does it,
scaled to the script arena (2 km x 1 km, 600 s, 6 s decisions), under the
presettings decided 2026-09-15:

* EQUAL SPEEDS: own ship and target run at the same constant cruise speed
  (default 6 m/s); the own ship starts at that speed and, unless
  speed_control=True, its actions are turn-only (port / straight /
  starboard at the same turn-rate limit as the target).  Overtaking and
  overtaken families are therefore excluded (neither ship can overtake).
* ROUTES: per episode a COLREG family (head_on, crossing_starboard,
  crossing_port, or 'mix' = one of them at random) is drawn as in v2:
  course difference, meeting time, closest-point-of-approach offset; the
  route is the straight chord start -> meeting point, extended past it, or
  shaped by a generated AIS trace sample when `trace` (an (n, T, 2) lon/lat
  .npy, e.g. data_prep output windows_lonlat.npy) is given.  Since 2026-09-24
  (mending plan part 1, step 4) a trace becomes a route through
  data_prep/ais_prep.py: C2 spline through the samples, tabulated every 5 m
  with heading and curvature, scaled by `trace_scale`, checked against the
  target's turn rate (fit_route_to_limit; traces that cannot meet it within
  50 m are left out), placed along its own start tangent, and played back
  from that table.  The old box-5 point smoothing is gone (it moved correctly
  timed routes by up to 215 m).
* TARGET (not an agent): by default automation='fixed' -- the target plays its
  route back exactly by arc length at cruise speed, independent of the
  attacker, so its track is fixed and identical on every replay.
  Optional reactive presets: steady course-holding ship (v3 behaviour): follows its route at
  cruise speed with a turn-rate limit and a capped cross-track correction;
  one COLREG alteration to starboard per emergency (DCPA / TCPA / range
  criterion held for a reaction latency), returns to the route when clear,
  then a cooldown before it may alter again.  automation='none' makes it
  inert, 'manual' / 'autonomous' scale the v2 presets to the arena.

Interface: ScenarioAttackEnv has the MassTestingEnv contract used by the
reference scripts (main_attack_ppo_scen.py), so PPO.py and viz_tool.py work
unchanged; evaluation() adds the scenario metadata.
"""
import copy
import math
import os
import sys

import numpy as np

import env_moving_obj as E
from env_moving_attack import MassTestingEnv as _AttackEnv

_DATA_PREP = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_prep')
if _DATA_PREP not in sys.path:
    sys.path.insert(0, _DATA_PREP)
import ais_prep                                  # noqa: E402  (route chain of the mending plan, part 1)

EARTH_R = 6371000.0
ROUTE_MAX_DEV = 50.0        # m at full size: largest move fit_route_to_limit may make to meet the yaw limit
ROUTE_POLYLINE_DS = 25.0    # m: spacing of the polyline kept next to a route table (drawing, reactive projection)

# families scaled to the arena: course difference [deg], meeting time [s], CPA offset [m]
SCENARIO_PARAMS = {
    'head_on':            dict(course=(174.0, 186.0), t_meet=(120.0, 300.0), dcpa=(-500.0, 500.0)),
    'crossing_starboard': dict(course=(60.0, 120.0),  t_meet=(120.0, 300.0), dcpa=(-500.0, 500.0)),
    'crossing_port':      dict(course=(-120.0, -60.0), t_meet=(120.0, 300.0), dcpa=(-500.0, 500.0)),
}
SCENARIOS = list(SCENARIO_PARAMS)

# target reaction presets (Ship_envre_v2 AUTOMATION_LEVELS scaled to the arena)
AUTOMATION = {
    'fixed':      dict(reactive=False, fixed_track=True),   # default: exact route playback, ignores the attacker
    'none':       dict(reactive=False, fixed_track=True),   # alias of 'fixed' (identical on straight routes)
    'manual':     dict(reactive=True, latency=20.0, alert_dcpa=250.0, alert_tcpa=120.0, evade_angle_deg=30.0,
                       detect_range=1500.0, cooldown=60.0, clear_time=10.0),
    'autonomous': dict(reactive=True, latency=0.0, alert_dcpa=300.0, alert_tcpa=150.0, evade_angle_deg=45.0,
                       detect_range=2000.0, cooldown=60.0, clear_time=10.0),
}


# ------------------------------------------------------------------ geometry
def cpa(p1, v1, p2, v2):
    """Distance and time of closest point of approach for constant velocities."""
    dp = np.asarray(p2, float) - np.asarray(p1, float)
    dv = np.asarray(v2, float) - np.asarray(v1, float)
    dv2 = float(dv @ dv)
    if dv2 < 1e-9:
        return float(np.linalg.norm(dp)), 0.0
    tcpa = -float(dp @ dv) / dv2
    dcpa = float(np.linalg.norm(dp + dv * tcpa))
    return dcpa, tcpa


def rotate_translate(trace, start, course_rad):
    """Rotate a trace so its initial course equals `course_rad`, then move its first point to `start`."""
    tr = np.asarray(trace, float) - np.asarray(trace, float)[0]
    seg = tr[min(3, len(tr) - 1)] - tr[0]
    a = course_rad - math.atan2(seg[1], seg[0])
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return tr @ R.T + np.asarray(start, float)


def extend_route(route, length, step=100.0):
    seg = route[-1] - route[-2]
    d = seg / max(np.linalg.norm(seg), 1e-9)
    n = int(math.ceil(length / step))
    ext = route[-1] + d[None, :] * (np.arange(1, n + 1) * step)[:, None]
    return np.concatenate([route, ext])


def route_length(route):
    return float(np.sum(np.linalg.norm(np.diff(route, axis=0), axis=1)))


def smooth_path(xy, window=5):
    """LEGACY (until 2026-09-24): box smoothing over `window` points.  No longer used by the environments;
    kept so that data_prep/check_mending_plan.py can still run the old chain for comparison."""
    xy = np.asarray(xy, float)
    if len(xy) < window + 2:
        return xy
    k = np.ones(window) / window
    pad = window // 2
    padded = np.concatenate([np.repeat(xy[:1], pad, 0), xy, np.repeat(xy[-1:], pad, 0)])
    return np.stack([np.convolve(padded[:, i], k, mode='valid') for i in range(2)], axis=1)


def load_trace_shapes(path):
    """(n, T, 2) lon/lat .npy -> list of xy traces in metres at full size, one tangent plane per trace
    (data_prep/ais_prep.ll_to_xy at the trace's first point), no smoothing."""
    arr = np.load(path)
    if arr.ndim == 2:
        arr = arr[None]
    shapes = []
    for tr in arr:
        ll = np.asarray(tr, float)
        ll = ll[np.all(np.isfinite(ll), axis=1)]
        if len(ll) > 3:
            shapes.append(ais_prep.ll_to_xy(ll[:, 0], ll[:, 1], float(ll[0, 0]), float(ll[0, 1])))
    if not shapes:
        raise ValueError('no usable traces in %s' % path)
    return shapes


def prepare_trace_routes(shapes, speed, yaw_limit_deg, scale=1.0, max_dev=ROUTE_MAX_DEV):
    """Trace shapes (metres, full size) -> route tables that the target can sail at `speed` under the
    yaw-rate limit once scaled by `scale` (ais_prep.fit_route_to_limit: the smallest spatial smoothing
    that meets the limit, rejected when it would move the route by more than max_dev metres).
    Returns (routes, info); each route carries its fit result under route['fit']."""
    routes, rejected, smooth, yaw = [], 0, [], []
    for xy in shapes:
        try:
            r, fit = ais_prep.fit_route_to_limit(np.asarray(xy, float), float(speed), float(yaw_limit_deg),
                                                 scale=float(scale), max_dev=float(max_dev))
        except ValueError:                      # too short for a route
            rejected += 1
            continue
        if not fit['ok']:
            rejected += 1
            continue
        r['fit'] = dict(fit)
        routes.append(r)
        smooth.append(fit['smooth_m'])
        yaw.append(fit['max_yaw_deg_s'])
    info = dict(n_traces=len(shapes), n_accepted=len(routes), n_rejected=rejected, speed=float(speed),
                yaw_limit_deg_s=float(yaw_limit_deg), scale=float(scale), max_dev_m=float(max_dev),
                smooth_m_max=float(max(smooth)) if smooth else 0.0,
                share_smoothed=float(np.mean(np.asarray(smooth) > 0)) if smooth else 0.0,
                max_yaw_deg_s=float(max(yaw)) if yaw else 0.0)
    if not routes:
        raise ValueError('none of the %d traces can be sailed at %.1f m/s under %.3f deg/s at scale %.3f '
                         '(moving them by more than %.0f m would be needed)' % (len(shapes), speed, yaw_limit_deg,
                                                                              scale, max_dev))
    return routes, info


def route_polyline(route, ds=ROUTE_POLYLINE_DS):
    """Points of a route table every `ds` metres (end point included), for drawing and projection."""
    k = max(1, int(round(ds / route['ds'])))
    xy = route['xy'][::k]
    return xy if np.array_equal(xy[-1], route['xy'][-1]) else np.concatenate([xy, route['xy'][-1:]])


def place_trace_route(shape, start, course_rad, run, scale=1.0):
    """A route table (or a raw trace, built without the limit check) scaled, turned onto `course_rad`
    along its own start tangent, moved to `start` and extended straight to at least `run` metres."""
    table = shape if isinstance(shape, dict) else ais_prep.build_route(np.asarray(shape, float))
    route = ais_prep.place_route(table, start, course_rad, scale=scale)
    if route['s'][-1] < run:
        route = ais_prep.extend_route(route, run - route['s'][-1])
    return route


def route_meta(route):
    fit = route.get('fit') if isinstance(route, dict) else None
    return dict(data_shaped=int(isinstance(route, dict)),
                route_smooth_m=round(float(fit['smooth_m']), 1) if fit else 0.0,
                route_max_yaw_deg_s=round(float(fit['max_yaw_deg_s']), 3) if fit else 0.0)


def make_scenario(name, own_pos, own_cog_deg, speed, rng, shape_trace=None, episode_time=600.0, trace_scale=1.0):
    """Target route + metadata for a family, equal speeds (v2 make_scenario scaled).

    shape_trace: None (straight chord), a route table from prepare_trace_routes, or a raw xy trace
    (built into a table without the yaw-limit check).  Returns route (xy polyline for drawing and the
    map) and route_table (the table the fixed-track target plays back) when a shape is given."""
    if name not in SCENARIO_PARAMS:
        raise ValueError('unknown scenario %r; choose %s or mix' % (name, SCENARIOS))
    p = SCENARIO_PARAMS[name]
    own_psi = math.radians(own_cog_deg)
    dpsi = math.radians(rng.uniform(*p['course']))
    psi_t = own_psi + dpsi
    t_meet = float(rng.uniform(*p['t_meet']))
    dcpa_off = float(rng.uniform(*p['dcpa']))
    d_own = np.array([math.cos(own_psi), math.sin(own_psi)])
    n_own = np.array([-d_own[1], d_own[0]])
    meet = np.asarray(own_pos, float) + d_own * speed * t_meet + n_own * dcpa_off
    d_t = np.array([math.cos(psi_t), math.sin(psi_t)])
    start = meet - d_t * speed * t_meet
    need = speed * (episode_time + 120.0)
    table = None
    if shape_trace is not None and len(shape_trace) > 3:
        table = place_trace_route(shape_trace, start, psi_t, need, scale=trace_scale)
        route = route_polyline(table)
    else:
        route = np.stack([start, meet])
        if route_length(route) < need:
            route = extend_route(route, need - route_length(route))
    r0 = float(np.linalg.norm(start - np.asarray(own_pos, float)))
    brg = math.degrees(math.atan2(start[1] - own_pos[1], start[0] - own_pos[0])) - own_cog_deg
    meta = dict(scenario=name, target_course_deg=round(math.degrees(psi_t) % 360.0, 1),
                course_diff_deg=round(math.degrees(dpsi), 1), t_meet_s=round(t_meet, 1),
                dcpa_offset_m=round(dcpa_off, 1), initial_range_m=round(r0, 1),
                initial_bearing_deg=round(E.wrap_deg(brg), 1), **route_meta(table))
    return dict(route=route, route_table=table, speed=speed, meta=meta)


# ------------------------------------------------------------------ target
class TargetShip:
    """Steady course-holding target ship on a route (obstacle-compatible interface)."""

    def __init__(self, route_xy, speed, L=200.0, W=50.0, risk_range=500.0, turn_rate_deg=1.0,
                 xte_gain_deg=2.0, xte_cap_deg=5.0, automation='fixed', id='T', meta=None):
        # route_xy: an (n, 2) polyline, or a route table (ais_prep.build_route / place_route) whose position
        # and heading the fixed track reads at arc length; the polyline kept next to it serves drawing and
        # the reactive presets' projection
        if isinstance(route_xy, dict):
            self.route_table = route_xy
            self.route = route_polyline(route_xy)
        else:
            self.route_table = None
            self.route = np.asarray(route_xy, float)
        self.cruise_speed = float(speed)
        self.L, self.W, self.risk_range = float(L), float(W), float(risk_range)
        self.turn_rate = float(turn_rate_deg)
        self.xte_gain, self.xte_cap = float(xte_gain_deg), float(xte_cap_deg)
        if automation not in AUTOMATION:
            raise ValueError('automation %r unknown; choose %s' % (automation, list(AUTOMATION)))
        self.automation = automation
        self.params = dict(AUTOMATION[automation])
        self.id = id
        self.meta = meta or {}
        self.reset()

    # --- obstacle-compatible interface ------------------------------------
    @property
    def is_moving(self):
        return True

    @property
    def sp(self):
        return self.speed

    @property
    def cog(self):
        return self.psi

    @property
    def Direction(self):
        return self.psi

    @property
    def velocity(self):
        rad = math.radians(self.psi)
        return self.speed * math.cos(rad), self.speed * math.sin(rad)

    def position(self, t=None):
        return self.x, self.y

    def corners(self, t=None):
        th = math.radians(self.psi)
        c, s = math.cos(th), math.sin(th)
        return [(self.x + u * c - v * s, self.y + u * s + v * c)
                for u, v in ((-self.L / 2, -self.W / 2), (self.L / 2, -self.W / 2),
                             (self.L / 2, self.W / 2), (-self.L / 2, self.W / 2))]

    def hull_distance(self, px, py, t=None):
        th = math.radians(self.psi)
        dx, dy = px - self.x, py - self.y
        u = dx * math.cos(th) + dy * math.sin(th)
        v = -dx * math.sin(th) + dy * math.cos(th)
        return math.hypot(max(abs(u) - self.L / 2, 0.0), max(abs(v) - self.W / 2, 0.0))

    # --- motion -----------------------------------------------------------
    def reset(self):
        keep = np.concatenate([[True], np.linalg.norm(np.diff(self.route, axis=0), axis=1) > 1e-9])
        self.route = self.route[keep]
        self._seg = np.diff(self.route, axis=0)
        self._cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(self._seg, axis=1))])
        self.s = 0.0                     # arc length travelled along the route [m]
        self.x, self.y = float(self.route[0][0]), float(self.route[0][1])
        if self.route_table is not None:
            self.psi = E.wrap_deg(math.degrees(float(self.route_table['psi'][0])))
        else:
            seg = self.route[1] - self.route[0]
            self.psi = math.degrees(math.atan2(seg[1], seg[0]))
        self.speed = self.cruise_speed
        self.evading = False
        self.evade_course = self.psi
        self.danger_time = 0.0
        self.clear_acc = 0.0
        self.cooldown_left = 0.0
        self.n_evasions = 0
        self.t = 0.0
        self.track = [(self.x, self.y)]
        self.psi_track = [self.psi]          # heading at every sub-step, for checks of the sailed yaw rate
        self.xte = 0.0

    def _project(self):
        """Nearest route segment: (segment course deg, signed cross-track m)."""
        p = np.array([self.x, self.y])
        best = (float('inf'), 0.0, 0.0)
        a, b = self.route[:-1], self.route[1:]
        seg = b - a
        seg_len2 = np.maximum(np.sum(seg ** 2, axis=1), 1e-9)
        frac = np.clip(np.sum((p - a) * seg, axis=1) / seg_len2, 0.0, 1.0)
        proj = a + seg * frac[:, None]
        dist = np.linalg.norm(proj - p, axis=1)
        i = int(np.argmin(dist))
        course = math.degrees(math.atan2(seg[i][1], seg[i][0]))
        # cross track: positive when the ship is left of the route direction
        cross = seg[i][0] * (p[1] - a[i][1]) - seg[i][1] * (p[0] - a[i][0])
        xte = float(cross / math.sqrt(seg_len2[i]))
        return course, xte

    def _route_course(self):
        course, xte = self._project()
        self.xte = xte
        corr = max(-self.xte_cap, min(self.xte_cap, -xte / 100.0 * self.xte_gain))
        return course + corr

    def _danger(self, own):
        p = self.params
        if own is None:
            return False, 0.0, 0.0
        rng = math.hypot(own.x - self.x, own.y - self.y)
        dcpa, tcpa = cpa((self.x, self.y), self.velocity, (own.x, own.y), own.velocity)
        danger = rng < p['detect_range'] and dcpa < p['alert_dcpa'] and 0.0 <= tcpa < p['alert_tcpa']
        return danger, dcpa, tcpa

    def advance(self, dt, own=None, t=None):
        p = self.params
        if p.get('fixed_track'):
            # fixed track: position is a function of time only, the attacker is ignored
            self.s += self.speed * dt
            if self.route_table is not None:
                # route table: position and heading of the spline at arc length s (the curvature that was
                # checked is the curvature that is sailed)
                pos, psi, _ = ais_prep.route_eval(self.route_table, self.s)
                self.x, self.y = float(pos[0]), float(pos[1])
                self.psi = E.wrap_deg(math.degrees(float(psi)))
            else:
                s = min(self.s, self._cum[-1])
                i = int(min(np.searchsorted(self._cum, s, side='right') - 1, len(self._seg) - 1))
                frac = (s - self._cum[i]) / max(self._cum[i + 1] - self._cum[i], 1e-9)
                self.x, self.y = (float(v) for v in self.route[i] + frac * self._seg[i])
                self.psi = math.degrees(math.atan2(self._seg[i][1], self._seg[i][0]))
            self.t += dt
            self.track.append((self.x, self.y))
            self.psi_track.append(self.psi)
            return
        if p.get('reactive'):
            danger, dcpa, tcpa = self._danger(own)
            if not self.evading:
                if danger and self.cooldown_left <= 0.0:
                    self.danger_time += dt
                    if self.danger_time >= p['latency']:
                        self.evading = True
                        self.evade_course = self.psi - p['evade_angle_deg']      # starboard = clockwise
                        self.n_evasions += 1
                        self.clear_acc = 0.0
                else:
                    self.danger_time = 0.0
            else:
                self.clear_acc = self.clear_acc + dt if not danger else 0.0
                if self.clear_acc >= p['clear_time']:
                    self.evading = False
                    self.danger_time = 0.0
                    self.cooldown_left = p['cooldown']
            self.cooldown_left = max(0.0, self.cooldown_left - dt)
        desired = self.evade_course if self.evading else self._route_course()
        err = E.wrap_deg(desired - self.psi)
        self.psi = E.wrap_deg(self.psi + max(-self.turn_rate * dt, min(self.turn_rate * dt, err)))
        rad = math.radians(self.psi)
        self.x += self.speed * math.cos(rad) * dt
        self.y += self.speed * math.sin(rad) * dt
        self.t += dt
        self.track.append((self.x, self.y))
        self.psi_track.append(self.psi)

    def __repr__(self):
        return 'TargetShip(%s, speed=%.1f, L=%g, W=%g, automation=%s)' % (
            self.meta.get('scenario', '?'), self.cruise_speed, self.L, self.W, self.automation)


# --------------------------------------------------------------------- env
class ScenarioAttackEnv(_AttackEnv):
    """Attack task against generated COLREG-family target ships, equal speeds."""

    per_episode_targets = True

    def __init__(self, own_ship, scenario='mix', cruise_speed=6.0, trace=None, automation='fixed',
                 cycle_scenarios=False, turn_rate_deg=1.0, speed_control=False, target_L=200.0,
                 target_W=50.0, risk_range=500.0, duration=60000, decision_interval=600,
                 reward_type='final_attack_reward', X_LEN=2000, Y_LEN=1000, save_dir='.',
                 attack_range=None, seed=0, trace_scale=1.0):
        # trace_scale: factor applied to trace shapes before they enter this hand-scaled arena (1.0 = the
        # full-size route; the encounter environment uses its standard's scale instead).  Traces whose
        # curvature at cruise speed exceeds turn_rate_deg after scaling, and cannot be brought under it
        # within 50 m, are left out (prepare_trace_routes).
        if scenario == 'mix':
            self.families = list(SCENARIOS)
        elif scenario in SCENARIO_PARAMS:
            self.families = [scenario]
        else:
            raise ValueError('scenario %r unknown; choose %s or mix' % (scenario, SCENARIOS))
        self.scenario_spec = scenario
        self.cruise = float(cruise_speed)
        self.trace_path = trace or None
        self.automation = automation
        self.cycle = bool(cycle_scenarios)
        self._cycle_i = 0
        self.turn_rate = float(turn_rate_deg)
        self.trace_scale = float(trace_scale)
        self.trace_shapes = load_trace_shapes(trace) if trace else None
        self.trace_routes, self.trace_info = (
            prepare_trace_routes(self.trace_shapes, self.cruise, self.turn_rate, scale=self.trace_scale)
            if self.trace_shapes else (None, None))
        self.speed_control = bool(speed_control)
        self.target_L, self.target_W, self.target_risk = float(target_L), float(target_W), float(risk_range)
        self.episode_time = (int(duration) // int(decision_interval)) * int(decision_interval) * E.TICK_S
        self.scen_rng = np.random.RandomState(seed)
        self.current = {}
        # own ship: starts at cruise speed, capped there unless speed control is on
        own = copy.deepcopy(own_ship)
        own.sp0 = self.cruise
        own.v_max = E.V_MAX if self.speed_control else self.cruise
        own.reset()
        R = self.turn_rate
        if self.speed_control:
            self.action_table = [(a, r) for a in (-E.A_MAX, 0.0, E.A_MAX) for r in (R, 0.0, -R)]
            self.action_names = list(E.ACTION_NAMES)
        else:
            self.action_table = [(0.0, R), (0.0, 0.0), (0.0, -R)]
            self.action_names = ['port', 'straight', 'starboard']
        self.target = None
        self._new_scenario(own)
        super().__init__(own, [self.target], [], None, duration=duration, decision_interval=decision_interval,
                         reward_type=reward_type, X_LEN=X_LEN, Y_LEN=Y_LEN, save_dir=save_dir,
                         attack_range=attack_range)
        self._cycle_i = 0          # constructor draws must not shift the evaluation cycle

    # ----------------------------------------------------------- scenarios
    def _new_scenario(self, own_proto=None):
        proto = own_proto if own_proto is not None else self.own_ship_proto
        if self.cycle:
            name = self.families[self._cycle_i % len(self.families)]
            self._cycle_i += 1
        else:
            name = self.families[self.scen_rng.randint(len(self.families))]
        shape = None
        if self.trace_routes:
            shape = self.trace_routes[self.scen_rng.randint(len(self.trace_routes))]
        scen = make_scenario(name, (proto.long0, proto.lat0), proto.cog0, self.cruise, self.scen_rng,
                             shape_trace=shape, episode_time=self.episode_time, trace_scale=self.trace_scale)
        route = scen['route_table'] if scen.get('route_table') is not None else scen['route']
        self.target = TargetShip(route, self.cruise, L=self.target_L, W=self.target_W,
                                 risk_range=self.target_risk, turn_rate_deg=self.turn_rate,
                                 automation=self.automation, id=name, meta=scen['meta'])
        self.current = dict(scen['meta'])
        self.ts_list, self.ob_list = [self.target], []
        self.objects, self.targets, self.hazards = [self.target], [self.target], []

    def seed(self, seed=None):
        super().seed(seed)
        self.scen_rng = np.random.RandomState(seed)
        return [seed]

    def reset(self):
        self._new_scenario()
        return super().reset()

    def target_track(self):
        return np.asarray(self.target.track)

    # ----------------------------------------------------------- reporting
    def evaluation(self):
        ev = super().evaluation()
        ev.update(self.current)
        ev.update(target_evasions=self.target.n_evasions, target_final_x=round(self.target.x, 1),
                  target_final_y=round(self.target.y, 1), cruise_speed=self.cruise, automation=self.automation)
        return ev

    def show_scenes(self, save_path=None, title=None):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon, Circle

        own = self.live_ownship
        path = np.asarray(own.path)
        tgt = self.target
        track = self.target_track()
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.add_patch(plt.Rectangle((0, -self.Y_LEN), self.X_LEN, 2 * self.Y_LEN, fill=False, ls='--', color='grey'))
        ax.plot(tgt.route[:, 0], tgt.route[:, 1], ':', color='tab:red', alpha=0.5, label='target route')
        ax.plot(track[:, 0], track[:, 1], '-', color='tab:red', lw=1.2, label='target track')
        c0 = TargetShip(tgt.route, tgt.cruise_speed, L=tgt.L, W=tgt.W, automation='none')
        ax.add_patch(Polygon(c0.corners(), closed=True, fc='tab:red', alpha=0.2, ec='tab:red'))
        ax.add_patch(Polygon(tgt.corners(), closed=True, fc='tab:red', alpha=0.7, ec='tab:red'))
        ax.add_patch(Circle((tgt.x, tgt.y), tgt.risk_range, fill=False, ls=':', color='tab:red'))
        if len(path):
            sc = ax.scatter(path[:, 0], path[:, 1], c=np.arange(len(path)), cmap='viridis', s=6)
            fig.colorbar(sc, ax=ax, label='sub-step (%.0f s each)' % self.dt_sub)
            ax.plot(path[0, 0], path[0, 1], 'k^', ms=8, label='own start')
            ax.plot(path[-1, 0], path[-1, 1], 'ks', ms=6, label='own end (%s)' % self.outcome)
        lim = max(self.X_LEN, 2 * self.Y_LEN)
        ax.set_xlim(-lim * 0.6, self.X_LEN + lim * 0.6)
        ax.set_ylim(-self.Y_LEN - lim * 0.6, self.Y_LEN + lim * 0.6)
        ax.set_aspect('equal'); ax.set_xlabel('long [m]'); ax.set_ylabel('lat [m]')
        m = self.current
        ax.set_title(title or '%s | ep %d | %s in %d decisions | dcpa off %.0f m, t_meet %.0f s, evasions %d'
                     % (m.get('scenario', '?'), self.episode, self.outcome, self.n_steps,
                        m.get('dcpa_offset_m', 0), m.get('t_meet_s', 0), tgt.n_evasions))
        ax.legend(loc='upper left', fontsize=8)
        if save_path is None:
            scene_dir = os.path.join(self.save_dir, 'scenes')
            os.makedirs(scene_dir, exist_ok=True)
            self._scene_count += 1
            save_path = os.path.join(scene_dir, 'scene_%s_ep%d_%d.png' % (m.get('scenario', 'scen'), self.episode,
                                                                          self._scene_count))
        fig.savefig(save_path, dpi=120)
        return fig

    show_path = show_scenes


__all__ = ['ScenarioAttackEnv', 'TargetShip', 'make_scenario', 'load_trace_shapes', 'prepare_trace_routes',
           'place_trace_route', 'route_polyline', 'route_meta', 'SCENARIOS', 'SCENARIO_PARAMS', 'AUTOMATION', 'cpa']
