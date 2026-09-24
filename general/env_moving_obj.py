"""
env_moving_obj.py -- world model and base MassTestingEnv (navigation task) of the scenario environments.

Rebuilt 2026-09-15 from the contract that the 2024 reference scripts main_ppo.py /
main_attack_ppo.py imposed on the module (the Feb-2024 original is not public; the
two scripts left the repository on 2026-09-24, their loop lives on in the attack
trainers).  env_moving_attack.py derives the attack task from this file.

World model
-----------
* Coordinates: x = `long` (0 .. X_LEN m), y = `lat` (-Y_LEN .. Y_LEN m).
  `cog` / `Direction` are degrees, 0 = +long, 90 = +lat, counter-clockwise
  positive.  All speeds `sp` are m/s.
* Time: `duration` and `decision_interval` are counted in 0.01 s ticks, so
  the script defaults (60000 / 600) give a 600 s episode with one decision
  every 6 s and at most 100 decisions.  Motion is integrated in 1 s
  sub-steps inside every decision, and collisions are checked per sub-step.
* Own ship: kinematic with rate limits (speed 0..V_MAX, acceleration
  +-A_MAX, turn rate +-R_MAX_DEG).  Discrete action = one of 9 combinations
  (decelerate / hold / accelerate) x (port / straight / starboard).
* Obstacles: rectangles L x W oriented by `Direction`, moving with constant
  velocity (`sp`, `cog`).  `risk_range` is a penalty zone around the hull;
  touching the hull (distance <= OWN_RADIUS) is a collision.  A target ship
  in `ts_list` is any object with the obstacle interface.
* Navigation task ends with success when the own ship is within
  `target_deviation_distance` of the navigation target with heading within
  `target_deviation_direction` degrees of the target direction; it fails on
  collision or leaving the map (+MAP_MARGIN); otherwise it times out.

Interface used by the scripts
-----------------------------
    env = MassTestingEnv(own_ship, ts_list, ob_list, nt, duration=..., decision_interval=...,
                         reward_type=..., X_LEN=..., Y_LEN=..., save_dir=...)
    env.observation_space.shape[0], env.action_space.n, env.seed(s)
    state = env.reset();  state, reward, done, success_flag = env.step(action)
    env.destination_step, env.show_scenes(), env.evaluation(), env.live_ownship.a, env.ownship_action

reward_type: 'final_step_reward' (terminal reward only, the original default),
'dense_step_reward' (progress shaping + risk penalty + terminal).  The attack
module accepts 'final_attack_reward' / 'dense_attack_reward'.
"""
import copy
import math
import os

import numpy as np

try:
    from gymnasium import spaces as _spaces
except Exception:  # pragma: no cover - gymnasium missing: minimal stand-ins
    _spaces = None

# ----------------------------------------------------------------- constants
TICK_S = 0.01          # one `duration` / `decision_interval` unit
SUBSTEP_S = 1.0        # integration and collision-check sub-step
V_MAX = 15.0           # m/s
A_MAX = 0.1            # m/s^2
R_MAX_DEG = 2.0        # deg/s
OWN_RADIUS = 10.0      # own ship treated as a disc of this radius (m)
MAP_MARGIN = 100.0     # allowed excursion beyond the map before "off map"
R_SUCCESS = 10.0
R_FAIL = -10.0
R_STEP = -0.01         # dense family only
R_RISK = -0.05         # dense family, per decision inside a risk zone (navigation)
PROGRESS_SCALE = 100.0  # dense family: reward = progress [m] / PROGRESS_SCALE

# action index -> (acceleration command m/s^2, turn-rate command deg/s)
ACTION_TABLE = [(a, r) for a in (-A_MAX, 0.0, A_MAX) for r in (R_MAX_DEG, 0.0, -R_MAX_DEG)]
ACTION_NAMES = ['%s/%s' % (a, r) for a in ('decel', 'hold', 'accel') for r in ('port', 'straight', 'starboard')]

REWARD_TYPES = ('final_step_reward', 'dense_step_reward', 'final_attack_reward', 'dense_attack_reward')


def wrap_deg(a):
    """Wrap an angle in degrees to (-180, 180]."""
    a = (a + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


class _Box:
    def __init__(self, low, high, shape, dtype=np.float32):
        self.low, self.high, self.shape, self.dtype = low, high, tuple(shape), dtype


class _Discrete:
    def __init__(self, n):
        self.n = int(n)
        self.shape = ()


def _make_box(dim):
    if _spaces is not None:
        return _spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)
    return _Box(-np.inf, np.inf, (dim,))


def _make_discrete(n):
    if _spaces is not None:
        return _spaces.Discrete(n)
    return _Discrete(n)


# ------------------------------------------------------------------- objects
class ownship:
    """Own ship state.  ownship(lat, long, sp, cog, a, rot): initial position,
    speed (m/s), course (deg), acceleration command (m/s^2) and turn-rate
    command (deg/s).  `a` and `rot` become per-decision command histories once
    the episode runs (main_*.py reads env.live_ownship.a)."""

    def __init__(self, lat=0.0, long=0.0, sp=0.0, cog=0.0, a=0.0, rot=0.0):
        self.lat0, self.long0, self.sp0, self.cog0 = float(lat), float(long), float(sp), float(cog)
        self.a0, self.rot0 = float(a), float(rot)
        self.v_max = V_MAX                        # per-instance speed cap (scenario envs lower it)
        self.reset()

    def reset(self):
        self.x, self.y, self.sp, self.cog = self.long0, self.lat0, self.sp0, self.cog0
        self.a_cmd, self.r_cmd = self.a0, self.rot0
        self.a, self.rot = [], []                 # command histories per decision
        self.path = [(self.x, self.y)]            # sub-step positions
        self.speeds = [self.sp]
        self.heading_offset = 0.0                 # course change since the start [deg] (for a course limit)

    def command(self, a_cmd, r_cmd):
        self.a_cmd, self.r_cmd = float(a_cmd), float(r_cmd)
        self.a.append(self.a_cmd)
        self.rot.append(self.r_cmd)

    def advance(self, dt):
        # optional per-instance limits (scenario envs set them): a_min/a_max, v_min, cog_limit
        a = self.a_cmd
        if getattr(self, 'a_min', None) is not None:
            a = min(max(a, self.a_min), self.a_max)
        self.sp = min(max(self.sp + a * dt, getattr(self, 'v_min', 0.0)), self.v_max)
        lim = getattr(self, 'cog_limit', None)
        off = self.heading_offset + self.r_cmd * dt
        if lim is not None:
            off = min(max(off, -lim), lim)
            self.cog = wrap_deg(self.cog0 + off)
        else:
            self.cog = wrap_deg(self.cog + self.r_cmd * dt)
        self.heading_offset = off
        rad = math.radians(self.cog)
        self.x += self.sp * math.cos(rad) * dt
        self.y += self.sp * math.sin(rad) * dt
        self.path.append((self.x, self.y))
        self.speeds.append(self.sp)

    @property
    def velocity(self):
        rad = math.radians(self.cog)
        return self.sp * math.cos(rad), self.sp * math.sin(rad)

    @property
    def lat(self):
        return self.y

    @property
    def long(self):
        return self.x


class obstacle:
    """Rectangular hull L x W (m) centred at (long, lat), oriented `Direction`
    (deg), moving with constant velocity `sp` (m/s) along `cog` (deg).
    `risk_range` (m) is the penalty / danger zone around the hull."""

    def __init__(self, id, lat, long, sp, cog, L, W, Direction, risk_range):
        self.id = id
        self.lat0, self.long0 = float(lat), float(long)
        self.sp, self.cog = float(sp), float(cog)
        self.L, self.W, self.Direction = float(L), float(W), float(Direction)
        self.risk_range = float(risk_range)

    @property
    def is_moving(self):
        return self.sp > 0.0

    @property
    def velocity(self):
        rad = math.radians(self.cog)
        return self.sp * math.cos(rad), self.sp * math.sin(rad)

    def position(self, t):
        vx, vy = self.velocity
        return self.long0 + vx * t, self.lat0 + vy * t

    def hull_distance(self, px, py, t):
        """Euclidean distance from a point to the hull rectangle (0 inside)."""
        cx, cy = self.position(t)
        th = math.radians(self.Direction)
        dx, dy = px - cx, py - cy
        u = dx * math.cos(th) + dy * math.sin(th)      # along the hull
        v = -dx * math.sin(th) + dy * math.cos(th)     # across the hull
        ex = max(abs(u) - self.L / 2.0, 0.0)
        ey = max(abs(v) - self.W / 2.0, 0.0)
        return math.hypot(ex, ey)

    def corners(self, t):
        cx, cy = self.position(t)
        th = math.radians(self.Direction)
        c, s = math.cos(th), math.sin(th)
        pts = []
        for u, v in ((-self.L / 2, -self.W / 2), (self.L / 2, -self.W / 2),
                     (self.L / 2, self.W / 2), (-self.L / 2, self.W / 2)):
            pts.append((cx + u * c - v * s, cy + u * s + v * c))
        return pts

    def __repr__(self):
        return ('obstacle(id=%r, lat=%g, long=%g, sp=%g, cog=%g, L=%g, W=%g, Direction=%g, risk_range=%g)'
                % (self.id, self.lat0, self.long0, self.sp, self.cog, self.L, self.W, self.Direction, self.risk_range))


class navigation_target:
    """Destination gate: reached when within `target_deviation_distance` (m)
    of (long, lat) with course within `target_deviation_direction` (deg) of
    `direction`."""

    def __init__(self, lat, long, direction, target_deviation_distance=200.0, target_deviation_direction=15.0):
        self.lat, self.long, self.direction = float(lat), float(long), float(direction)
        self.target_deviation_distance = float(target_deviation_distance)
        self.target_deviation_direction = float(target_deviation_direction)

    @property
    def x(self):
        return self.long

    @property
    def y(self):
        return self.lat

    def distance(self, px, py):
        return math.hypot(px - self.long, py - self.lat)

    def heading_error(self, cog):
        return wrap_deg(cog - self.direction)

    def reached(self, px, py, cog):
        return (self.distance(px, py) <= self.target_deviation_distance
                and abs(self.heading_error(cog)) <= self.target_deviation_direction)

    def __repr__(self):
        return 'navigation_target(lat=%g, long=%g, direction=%g, dist=%g, dir=%g)' % (
            self.lat, self.long, self.direction, self.target_deviation_distance, self.target_deviation_direction)


# ----------------------------------------------------------------- the env
class MassTestingEnv:
    """Discrete-action MASS testing scene.  task='navigate' (this module) or
    'attack' (env_moving_attack.MassTestingEnv)."""

    metadata = {'render.modes': ['human']}
    OUTCOMES = ('running', 'success', 'collision', 'off_map', 'timeout')

    def __init__(self, own_ship, ts_list, ob_list, nt, duration=60000, decision_interval=600,
                 reward_type='final_step_reward', X_LEN=2000, Y_LEN=1000, save_dir='.',
                 task='navigate', attack_range=None, init_noise=0.0):
        if task not in ('navigate', 'attack'):
            raise ValueError('task must be navigate or attack, got %r' % task)
        if reward_type not in REWARD_TYPES:
            raise ValueError('reward_type %r unknown; choose one of %s' % (reward_type, REWARD_TYPES))
        self.task = task
        self.reward_family = 'final' if reward_type.startswith('final') else 'dense'
        self.reward_type = reward_type
        self.own_ship_proto = own_ship
        self.ts_list = list(ts_list or [])
        self.ob_list = list(ob_list or [])
        self.objects = self.ob_list + self.ts_list
        self.nt = nt
        self.duration = int(duration)
        self.decision_interval = int(decision_interval)
        self.X_LEN, self.Y_LEN = float(X_LEN), float(Y_LEN)
        self.save_dir = save_dir
        self.attack_range = attack_range
        self.init_noise = float(init_noise)

        self.dt_decision = self.decision_interval * TICK_S
        self.n_sub = max(1, int(round(self.dt_decision / SUBSTEP_S)))
        self.dt_sub = self.dt_decision / self.n_sub
        self.max_decisions = max(1, self.duration // self.decision_interval)
        self.diag = math.hypot(self.X_LEN, 2.0 * self.Y_LEN)

        # attack targets: declared target ships, else every moving obstacle
        if self.task == 'attack':
            self.targets = self.ts_list if self.ts_list else [o for o in self.ob_list if o.is_moving]
            if not self.targets:
                raise ValueError('attack task needs a target ship (ts_list) or a moving obstacle')
        else:
            self.targets = []
        self.hazards = [o for o in self.objects if o not in self.targets]

        self.n_obj = max(1, len(self.objects))
        self.obs_dim = 11 + 8 * self.n_obj
        self.observation_space = _make_box(self.obs_dim)
        if not getattr(self, 'action_table', None):
            self.action_table = list(ACTION_TABLE)
            self.action_names = list(ACTION_NAMES)
        self.action_space = _make_discrete(len(self.action_table))
        self.rng = np.random.RandomState(0)
        self.episode = 0
        self._episode_started = False
        self._scene_count = 0
        self.live_ownship = copy.deepcopy(own_ship)
        self.ownship_action = []
        self.reset()

    # ------------------------------------------------------------ gym api
    def seed(self, seed=None):
        self.rng = np.random.RandomState(seed)
        return [seed]

    def reset(self):
        self.live_ownship = copy.deepcopy(self.own_ship_proto)
        self.live_ownship.reset()
        if self.init_noise > 0:
            self.live_ownship.x += self.rng.uniform(-self.init_noise, self.init_noise)
            self.live_ownship.y += self.rng.uniform(-self.init_noise, self.init_noise)
        self.t = 0.0
        self.n_steps = 0
        self.destination_step = 0
        self.outcome = 'running'
        self.done = False
        self.ownship_action = []
        self.rewards = []
        self.risk_steps = 0
        self.min_hazard_dist = float('inf')
        self.min_target_dist = float('inf')
        self.total_reward = 0.0
        self._episode_started = False
        for o in self.objects:
            rs = getattr(o, 'reset', None)
            if rs is not None:
                rs()
        self.d_init = self._goal_distance()
        self.d_prev = self.d_init
        return self._observe()

    def step(self, action):
        if self.done:
            raise RuntimeError('step() called on a finished episode; call reset()')
        if not self._episode_started:
            self.episode += 1
            self._episode_started = True
        action = int(action)
        a_cmd, r_cmd = self.action_table[action]
        own = self.live_ownship
        own.command(a_cmd, r_cmd)
        self.ownship_action.append(action)

        collided = off_map = attacked = False
        for _ in range(self.n_sub):
            own.advance(self.dt_sub)
            self.t += self.dt_sub
            for o in self.objects:
                adv = getattr(o, 'advance', None)
                if adv is not None:
                    adv(self.dt_sub, own, self.t)
            for o in self.hazards:
                d = o.hull_distance(own.x, own.y, self.t)
                self.min_hazard_dist = min(self.min_hazard_dist, d)
                if d <= OWN_RADIUS:
                    collided = True
            for o in self.targets:
                d = o.hull_distance(own.x, own.y, self.t)
                self.min_target_dist = min(self.min_target_dist, d)
                if d <= self._attack_range(o):
                    attacked = True
            if (own.x < -MAP_MARGIN or own.x > self.X_LEN + MAP_MARGIN
                    or abs(own.y) > self.Y_LEN + MAP_MARGIN):
                off_map = True
            if collided or off_map or attacked:
                break
        self.n_steps += 1

        in_risk = any(o.hull_distance(own.x, own.y, self.t) <= o.risk_range for o in self.objects)
        if in_risk:
            self.risk_steps += 1

        if self.task == 'navigate':
            success = self.nt.reached(own.x, own.y, own.cog)
        else:
            success = attacked
        if collided:
            self.outcome = 'collision'
        elif off_map:
            self.outcome = 'off_map'
        elif success:
            self.outcome = 'success'
        elif self.n_steps >= self.max_decisions:
            self.outcome = 'timeout'
        self.done = self.outcome != 'running'

        if self.outcome == 'success':
            success_flag = 1
        elif self.outcome in ('collision', 'off_map'):
            success_flag = -1
        else:
            success_flag = 0

        d_now = self._goal_distance()
        reward = self._reward(d_now, in_risk)
        self.d_prev = d_now
        self.rewards.append(reward)
        self.total_reward += reward
        if self.done:
            self.destination_step = self.n_steps
        return self._observe(), reward, self.done, success_flag

    def close(self):
        pass

    def render(self, mode='human'):
        return self.show_scenes()

    # --------------------------------------------------------- internals
    def _attack_range(self, target):
        if self.attack_range is not None:
            return float(self.attack_range)
        return target.L                      # one ship length: close-quarters criterion

    def _goal_distance(self):
        own = self.live_ownship
        if self.task == 'navigate':
            return self.nt.distance(own.x, own.y)
        return min(o.hull_distance(own.x, own.y, self.t) for o in self.targets)

    def _reward(self, d_now, in_risk):
        if self.reward_family == 'final':
            if self.outcome == 'success':
                return R_SUCCESS
            if self.outcome in ('collision', 'off_map'):
                return R_FAIL
            if self.outcome == 'timeout':
                ref = self.min_target_dist if self.task == 'attack' else d_now
                return R_FAIL * min(1.0, ref / max(self.d_init, 1.0))
            return 0.0
        # dense family: progress shaping every decision
        reward = (self.d_prev - d_now) / PROGRESS_SCALE + R_STEP
        if in_risk:
            reward += -R_RISK if self.task == 'attack' else R_RISK
        if self.outcome == 'success':
            reward += R_SUCCESS
        elif self.outcome in ('collision', 'off_map'):
            reward += R_FAIL
        return reward

    def _observe(self):
        own = self.live_ownship
        rad = math.radians(own.cog)
        vx, vy = own.velocity
        obs = [own.x / self.X_LEN, own.y / self.Y_LEN, own.sp / V_MAX, math.cos(rad), math.sin(rad)]
        # goal block: navigation target (navigate) or nearest target ship (attack)
        if self.task == 'navigate':
            gx, gy = self.nt.x, self.nt.y
            herr = math.radians(self.nt.heading_error(own.cog))
        else:
            tgt = min(self.targets, key=lambda o: o.hull_distance(own.x, own.y, self.t))
            gx, gy = tgt.position(self.t)
            herr = math.radians(wrap_deg(own.cog - tgt.cog))
        dx, dy = gx - own.x, gy - own.y
        dist = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx) - rad
        obs += [dx / self.X_LEN, dy / self.Y_LEN, dist / self.diag, math.cos(bearing), math.sin(bearing), herr / math.pi]
        for i in range(self.n_obj):
            if i < len(self.objects):
                o = self.objects[i]
                ox, oy = o.position(self.t)
                ovx, ovy = o.velocity
                dx, dy = ox - own.x, oy - own.y
                dh = o.hull_distance(own.x, own.y, self.t)
                bearing = math.atan2(dy, dx) - rad
                obs += [dx / self.X_LEN, dy / self.Y_LEN, dh / self.diag, math.cos(bearing), math.sin(bearing),
                        (ovx - vx) / V_MAX, (ovy - vy) / V_MAX, 1.0 if dh <= o.risk_range else 0.0]
            else:
                obs += [0.0] * 8
        return np.asarray(obs, dtype=np.float32)

    # --------------------------------------------------------- reporting
    def evaluation(self):
        """Fixed-key episode record (main_*.py writes it with csv.DictWriter)."""
        own = self.live_ownship
        path = np.asarray(own.path)
        seg = np.diff(path, axis=0) if len(path) > 1 else np.zeros((0, 2))
        a_hist = np.asarray(own.a) if own.a else np.zeros(0)
        r_hist = np.asarray(own.rot) if own.rot else np.zeros(0)
        return dict(
            episode=self.episode,
            task=self.task,
            reward_type=self.reward_type,
            steps=self.n_steps,
            sim_time_s=round(self.t, 1),
            outcome=self.outcome,
            success=int(self.outcome == 'success'),
            final_x=round(own.x, 1),
            final_y=round(own.y, 1),
            final_speed=round(own.sp, 2),
            final_cog_deg=round(own.cog, 1),
            dist_to_destination=round(self.nt.distance(own.x, own.y), 1) if self.nt is not None else -1.0,
            heading_error_deg=round(self.nt.heading_error(own.cog), 1) if self.nt is not None else 0.0,
            min_hazard_dist=round(self.min_hazard_dist, 1) if self.hazards else -1.0,
            min_target_dist=round(self.min_target_dist, 1) if self.targets else -1.0,
            steps_in_risk=self.risk_steps,
            path_length=round(float(np.hypot(seg[:, 0], seg[:, 1]).sum()) if len(seg) else 0.0, 1),
            mean_speed=round(float(np.mean(own.speeds)), 2),
            n_accel_cmds=int((a_hist > 0).sum()),
            n_decel_cmds=int((a_hist < 0).sum()),
            n_turn_cmds=int((r_hist != 0).sum()),
            total_reward=round(self.total_reward, 3),
        )

    def show_scenes(self, save_path=None, title=None):
        """Draw the scene (map, hulls at start and end, risk zones, own path,
        destination gate) and save it under save_dir/scenes/ unless save_path
        is given.  Never calls plt.show() (the scripts use the pdf backend)."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon, Circle

        own = self.live_ownship
        path = np.asarray(own.path)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.add_patch(plt.Rectangle((0, -self.Y_LEN), self.X_LEN, 2 * self.Y_LEN, fill=False, ls='--', color='grey'))
        for o in self.objects:
            is_t = o in self.targets
            colour = 'tab:red' if is_t else 'tab:brown'
            ax.add_patch(Polygon(o.corners(0.0), closed=True, fc=colour, alpha=0.25, ec=colour))
            ax.add_patch(Circle(o.position(0.0), o.risk_range, fill=False, ls=':', color=colour, alpha=0.6))
            if o.is_moving:
                x0, y0 = o.position(0.0)
                x1, y1 = o.position(self.t)
                ax.plot([x0, x1], [y0, y1], ':', color=colour)
                ax.add_patch(Polygon(o.corners(self.t), closed=True, fc=colour, alpha=0.6, ec=colour))
                ax.add_patch(Circle((x1, y1), o.risk_range, fill=False, ls=':', color=colour))
            ax.annotate('%s %s' % ('target' if is_t else 'obst', o.id), o.position(self.t), fontsize=8, color=colour)
        if self.nt is not None and self.task == 'navigate':
            ax.add_patch(Circle((self.nt.x, self.nt.y), self.nt.target_deviation_distance, fill=False, color='tab:green'))
            rad = math.radians(self.nt.direction)
            ax.arrow(self.nt.x, self.nt.y, 150 * math.cos(rad), 150 * math.sin(rad),
                     head_width=40, color='tab:green')
        if len(path):
            sc = ax.scatter(path[:, 0], path[:, 1], c=np.arange(len(path)), cmap='viridis', s=6)
            fig.colorbar(sc, ax=ax, label='sub-step (%.0f s each)' % self.dt_sub)
            ax.plot(path[0, 0], path[0, 1], 'k^', ms=8, label='own start')
            ax.plot(path[-1, 0], path[-1, 1], 'ks', ms=6, label='own end (%s)' % self.outcome)
        ax.set_xlim(-MAP_MARGIN - 200, self.X_LEN + MAP_MARGIN + 200)
        ax.set_ylim(-self.Y_LEN - MAP_MARGIN - 200, self.Y_LEN + MAP_MARGIN + 200)
        ax.set_aspect('equal')
        ax.set_xlabel('long [m]')
        ax.set_ylabel('lat [m]')
        ax.set_title(title or '%s | episode %d | %s in %d decisions (%.0f s) | return %.2f'
                     % (self.task, self.episode, self.outcome, self.n_steps, self.t, self.total_reward))
        ax.legend(loc='upper left', fontsize=8)
        if save_path is None:
            scene_dir = os.path.join(self.save_dir, 'scenes')
            os.makedirs(scene_dir, exist_ok=True)
            self._scene_count += 1
            save_path = os.path.join(scene_dir, 'scene_%s_ep%d_%d.png' % (self.task, self.episode, self._scene_count))
        fig.savefig(save_path, dpi=120)
        return fig

    show_path = show_scenes

    def show_parameters(self):
        print('task=%s reward_type=%s decisions<=%d dt=%.1fs objects=%d targets=%d nt=%r'
              % (self.task, self.reward_type, self.max_decisions, self.dt_decision,
                 len(self.objects), len(self.targets), self.nt))


__all__ = ['MassTestingEnv', 'ownship', 'navigation_target', 'obstacle', 'ACTION_TABLE', 'ACTION_NAMES']
