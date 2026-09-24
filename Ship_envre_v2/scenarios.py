"""
Encounter scenarios defined in COLREG terms.

Each scenario places the target ship relative to the own ship's initial
state so that the encounter geometry matches the rule that governs it:

  head_on             Rule 14: reciprocal courses (+-6 deg), target ahead
  crossing_starboard  Rule 15: target crosses from the own ship's starboard
                      side, own ship is the give-way vessel
  crossing_port       Rule 15: target crosses from port, own ship stands on
  overtaking          Rule 13: own ship approaches a slower target from more
                      than 22.5 deg abaft its beam
  overtaken           Rule 13: a faster target approaches from astern
  data                the generated merge line as recorded (reversed):
                      parallel-lane / head-on geometry of the data set

Geometry: a meeting point is placed T_meet ahead on the own ship's course,
offset laterally by the intended DCPA; the target starts T_meet earlier on
its own course through that point.  The target's ROUTE SHAPE still comes
from a generated merge-line sample, rotated so that its initial course is
the scenario course and translated to the start point, so the generated
data stay in the loop; `shape='straight'` gives a straight route instead.

Angles are radians, counter-clockwise from +x (math convention), so a
"starboard" alteration is negative.
"""
import numpy as np

NM = 1852.0

SCENARIOS = ['head_on', 'crossing_starboard', 'crossing_port', 'overtaking', 'overtaken', 'data']

# parameter ranges per scenario (uniform sampling)
SCENARIO_PARAMS = {
    #                     course diff [deg]   T_meet [min]  dcpa offset [nm]  target speed rule
    'head_on':           dict(course=(174, 186), t_meet=(15, 30), dcpa=(-0.3, 0.3), speed=('abs', 4.0, 9.0)),
    'crossing_starboard': dict(course=(60, 120), t_meet=(15, 30), dcpa=(-0.3, 0.3), speed=('abs', 4.0, 9.0)),
    'crossing_port':     dict(course=(-120, -60), t_meet=(15, 30), dcpa=(-0.3, 0.3), speed=('abs', 4.0, 9.0)),
    'overtaking':        dict(course=(-15, 15), t_meet=(12, 25), dcpa=(-0.3, 0.3), speed=('ratio', 0.4, 0.65)),
    'overtaken':         dict(course=(-15, 15), t_meet=(12, 25), dcpa=(-0.3, 0.3), speed=('ratio', 1.4, 2.2)),
}
# own-ship speed ranges that make the scenario physically possible
OWN_SPEED = {'overtaking': (8.0, 11.0), 'overtaken': (3.0, 5.0)}


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rotate_translate(trace, start, course):
    """Rotate a trace so its initial course equals `course`, then move its
    first point to `start`."""
    tr = np.asarray(trace, float) - trace[0]
    seg = tr[min(3, len(tr) - 1)] - tr[0]
    c0 = np.arctan2(seg[1], seg[0])
    a = course - c0
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return tr @ R.T + np.asarray(start, float)


def extend_route(route, length, step=100.0):
    """Extend a route beyond its last point along its final course."""
    seg = route[-1] - route[-2]
    d = seg / max(np.linalg.norm(seg), 1e-9)
    n = int(np.ceil(length / step))
    ext = route[-1] + d[None, :] * (np.arange(1, n + 1) * step)[:, None]
    return np.concatenate([route, ext])


def make_scenario(name, own_pos, own_psi, own_speed, rng, shape_trace=None, min_len=8000.0):
    """Return dict(route, speed, meta) for the target ship."""
    if name not in SCENARIO_PARAMS:
        raise ValueError('unknown scenario %s' % name)
    p = SCENARIO_PARAMS[name]
    dpsi = np.radians(rng.uniform(*p['course']))
    psi_t = wrap_pi(own_psi + dpsi)
    t_meet = rng.uniform(*p['t_meet']) * 60.0
    dcpa_off = rng.uniform(*p['dcpa']) * NM
    kind, lo, hi = p['speed']
    v_t = rng.uniform(lo, hi) if kind == 'abs' else float(np.clip(own_speed * rng.uniform(lo, hi), 1.5, 12.0))

    d_own = np.array([np.cos(own_psi), np.sin(own_psi)])
    n_own = np.array([-d_own[1], d_own[0]])
    meet = np.asarray(own_pos, float) + d_own * own_speed * t_meet + n_own * dcpa_off
    d_t = np.array([np.cos(psi_t), np.sin(psi_t)])
    start = meet - d_t * v_t * t_meet

    if shape_trace is not None and len(shape_trace) > 3:
        route = rotate_translate(shape_trace, start, psi_t)
    else:
        route = np.stack([start, meet])
    # make sure the route runs well past the meeting point
    route_len = float(np.sum(np.linalg.norm(np.diff(route, axis=0), axis=1)))
    need = v_t * t_meet * 2.0 + 4 * NM
    if route_len < max(need, min_len):
        route = extend_route(route, max(need, min_len) - route_len)
    r0 = float(np.linalg.norm(start - own_pos))
    brg = wrap_pi(np.arctan2(start[1] - own_pos[1], start[0] - own_pos[0]) - own_psi)
    meta = dict(scenario=name, target_course_deg=float(np.degrees(psi_t)), course_diff_deg=float(np.degrees(dpsi)),
                t_meet_min=t_meet / 60.0, dcpa_offset_nm=dcpa_off / NM, target_speed=v_t,
                initial_range_nm=r0 / NM, initial_bearing_deg=float(np.degrees(brg)))
    return dict(route=route, speed=v_t, meta=meta)


def own_speed_range(name, default):
    return OWN_SPEED.get(name, default)


def colreg_role(name):
    """Own ship's role under the rules for the scenario (for compliance checks)."""
    return {'head_on': 'both_give_way', 'crossing_starboard': 'give_way', 'crossing_port': 'stand_on',
            'overtaking': 'give_way', 'overtaken': 'stand_on', 'data': 'unknown'}.get(name, 'unknown')
