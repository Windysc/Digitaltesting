"""
encounter_standard.py -- the collision standard of the earlier Ship_envre_v2 / v3
builds, ported to the scenario environments of this folder (2026-09-15).

Sources (the v2 / v3 builds, in the repository history up to commit 3b9feb0):
  Ship_envre_v2/encounter.py   DCPA / TCPA, ship domain, CRI, severity 0-3, COLREG type
  Ship_envre_v2/grading.py     event grade 0-3 and event score 0-100
  Ship_envre_v2/scenarios.py   encounter geometry (course difference, meeting time, DCPA offset)
  Ship_envre_v3/ship_env_v3.py encounter lifecycle (approach / action / passing / clear)

Severity (v2 encounter.py, unchanged):
  0 clear
  1 CPA warning      DCPA < d_safe and 0 <= TCPA < t_safe
  2 domain violation the other ship is inside either ship's domain ellipse
                     (semi-axes domain_a = 4 L along the course, domain_b = 1.6 L across)
  3 collision        centre distance < collision_L * L   (1.0 L)

Lifecycle (v3 ship_env_v3._update_phase, unchanged):
  phase 0 approach   CPA ahead, no warning yet
  phase 1 action     a CPA warning or higher severity is / was active
  phase 2 passing    CPA behind (TCPA < 0, or no relative motion) after phase 1
  phase 3 clear      severity 0, range above r_clear and opening for n_clear
                     consecutive decisions (a second approach resets the count)

Scale.  v2 fixes the values for its tanker (L = 244.74 m): d_safe 926 m
(0.5 nm), t_safe 900 s, r_clear 3704 m (2 nm), n_clear 300 s, meeting time
15-30 min, turning radius about 5 L, CRI distance 2000 m.  Every length and
time here is v2 x `scale` with the speed kept (a kinematic model has no other
invariant), so all ratios of the standard are preserved:

  scale 'maritime' = 1.0      the v2 numbers as they are (L 244.7 m)
  scale 'arena'    = 35/244.74 = 0.143  the 2 km arena of the reference scripts
                                (L 35 m, d_safe 132 m, t_safe 129 s, domain 140 x 56 m,
                                 collision 35 m, meeting time 129-257 s, r_clear 530 m)
  a number                    any other factor
The hull aspect W = 0.25 L is the reference scripts' 200 x 50 m ship.
"""
import math

import numpy as np

NM = 1852.0

# the Ship_envre_v2 values, for its L = 244.74 m ship
V2_REF = dict(L=244.74, d_safe=926.0, t_safe=900.0, domain_a_L=4.0, domain_b_L=1.6, collision_L=1.0,
              cri_dr=2000.0, cri_al=0.2, r_clear=2.0 * NM, n_clear_s=300.0, t_meet_s=(900.0, 1800.0),
              v2_dcpa_offset=0.3 * NM, turn_radius_L=5.0, time_in_domain_norm=300.0, closing_norm=10.0,
              success_severity=2, hold_steps=2, cap_over_meeting=6.0)
SCALES = {'maritime': 1.0, 'arena': 35.0 / 244.74}

SEVERITY_NAMES = {0: 'clear', 1: 'cpa_warning', 2: 'domain_violation', 3: 'collision'}
GRADE_NAMES = {0: 'safe_passage', 1: 'close_quarters', 2: 'domain_infringement', 3: 'collision'}
PHASE_NAMES = {0: 'approach', 1: 'action', 2: 'passing', 3: 'clear'}


def resolve_scale(scale):
    if isinstance(scale, str):
        if scale in SCALES:
            return scale, SCALES[scale]
        try:
            return 'x%s' % scale, float(scale)
        except ValueError:
            raise ValueError('scale %r unknown; choose %s or a number' % (scale, sorted(SCALES)))
    return 'x%g' % scale, float(scale)


class CollisionStandard:
    """All thresholds of the v2 / v3 standard at one scale."""

    def __init__(self, scale='arena', speed=6.0, d_safe=None, t_safe=None, domain_a_L=None, domain_b_L=None,
                 collision_L=None, r_clear=None, n_clear_s=None, success_severity=None, hold_steps=None):
        self.scale_name, s = resolve_scale(scale)
        R = V2_REF
        self.scale = s
        self.speed = float(speed)
        self.L = R['L'] * s
        self.W = 0.25 * self.L
        self.domain_a_L = float(domain_a_L if domain_a_L is not None else R['domain_a_L'])
        self.domain_b_L = float(domain_b_L if domain_b_L is not None else R['domain_b_L'])
        self.collision_L = float(collision_L if collision_L is not None else R['collision_L'])
        self.domain_a = self.domain_a_L * self.L
        self.domain_b = self.domain_b_L * self.L
        self.collision_dist = self.collision_L * self.L
        self.d_safe = float(d_safe) if d_safe else R['d_safe'] * s
        self.t_safe = float(t_safe) if t_safe else R['t_safe'] * s
        self.cri_dr = R['cri_dr'] * s
        self.cri_al = R['cri_al']
        self.r_clear = float(r_clear) if r_clear else R['r_clear'] * s
        self.n_clear_s = float(n_clear_s) if n_clear_s else R['n_clear_s'] * s
        self.t_meet = (R['t_meet_s'][0] * s, R['t_meet_s'][1] * s)
        self.v2_dcpa_offset = R['v2_dcpa_offset'] * s
        self.turn_rate_deg = math.degrees(self.speed / (R['turn_radius_L'] * self.L))
        self.time_in_domain_norm = R['time_in_domain_norm'] * s
        self.closing_norm = R['closing_norm']
        self.success_severity = int(success_severity if success_severity is not None else R['success_severity'])
        self.hold_steps = int(hold_steps if hold_steps is not None else R['hold_steps'])
        self.cap_s = R['cap_over_meeting'] * self.t_meet[1]

    def as_dict(self):
        keys = ['scale_name', 'scale', 'speed', 'L', 'W', 'd_safe', 't_safe', 'domain_a', 'domain_b', 'collision_dist',
                'cri_dr', 'cri_al', 'r_clear', 'n_clear_s', 't_meet', 'v2_dcpa_offset', 'turn_rate_deg',
                'time_in_domain_norm', 'closing_norm', 'success_severity', 'hold_steps', 'cap_s']
        return {k: getattr(self, k) for k in keys}

    def __repr__(self):
        return ('CollisionStandard(%s x%.3f: L %.0f m, d_safe %.0f m, t_safe %.0f s, domain %.0f x %.0f m, '
                'collision %.0f m, r_clear %.0f m / %.0f s, meeting %.0f-%.0f s, turn %.2f deg/s)' % (
                    self.scale_name, self.scale, self.L, self.d_safe, self.t_safe, self.domain_a, self.domain_b,
                    self.collision_dist, self.r_clear, self.n_clear_s, self.t_meet[0], self.t_meet[1],
                    self.turn_rate_deg))


# ------------------------------------------------------------------ criteria (v2 encounter.py)
def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def cpa(p_own, v_own, p_other, v_other):
    """DCPA [m], TCPA [s].  TCPA < 0: CPA passed, DCPA = current distance; no relative motion: TCPA = inf."""
    r = np.asarray(p_other, float) - np.asarray(p_own, float)
    vr = np.asarray(v_other, float) - np.asarray(v_own, float)
    vr2 = float(vr @ vr)
    dist = float(math.hypot(r[0], r[1]))
    if vr2 < 1e-9:
        return dist, math.inf
    tcpa = -float(r @ vr) / vr2
    if tcpa < 0:
        return dist, tcpa
    return float(np.linalg.norm(r + vr * tcpa)), tcpa


def domain_margin(p_point, p_ship, psi_ship, a, b):
    """Normalised ellipse radius of p_point in the ship's domain frame (< 1 inside)."""
    dx, dy = p_point[0] - p_ship[0], p_point[1] - p_ship[1]
    c, s = math.cos(psi_ship), math.sin(psi_ship)
    xb = c * dx + s * dy
    yb = -s * dx + c * dy
    return math.sqrt((xb / a) ** 2 + (yb / b) ** 2)


def collision_risk_index(dcpa, tcpa, speed, std):
    if not math.isfinite(tcpa) or tcpa < 0:
        return 0.0
    return float(np.clip(std.cri_al ** ((dcpa + speed * tcpa) / std.cri_dr), 0.0, 1.0))


def severity(dist, dcpa, tcpa, inside_domain, std):
    if dist < std.collision_dist:
        return 3
    if inside_domain:
        return 2
    if dcpa < std.d_safe and 0.0 <= tcpa < std.t_safe:
        return 1
    return 0


def encounter_type(psi_own, psi_other, p_own, p_other, rel_speed=None, speed=None):
    """COLREG situation seen from own (v2), plus 'parallel' for equal-speed same-course lanes."""
    dx, dy = p_other[0] - p_own[0], p_other[1] - p_own[1]
    brg_from_own = wrap_pi(math.atan2(dy, dx) - psi_own)
    brg_from_other = wrap_pi(math.atan2(-dy, -dx) - psi_other)
    course_diff = abs(wrap_pi(psi_other - psi_own))
    if rel_speed is not None and speed and rel_speed < 0.1 * speed and course_diff < math.radians(15):
        return 'parallel'
    if course_diff > math.radians(174) and abs(brg_from_own) < math.radians(6):
        return 'head_on'
    if abs(brg_from_other) > math.radians(112.5):
        return 'overtaking'
    if abs(brg_from_own) > math.radians(112.5):
        return 'overtaken'
    return 'crossing_starboard' if brg_from_own < 0 else 'crossing_port'


def assess(p_own, v_own, psi_own, p_other, v_other, psi_other, std):
    """Full assessment from own's point of view (v2 encounter.assess)."""
    dist = math.hypot(p_other[0] - p_own[0], p_other[1] - p_own[1])
    dcpa, tcpa = cpa(p_own, v_own, p_other, v_other)
    m_other = domain_margin(p_own, p_other, psi_other, std.domain_a, std.domain_b)   # own inside the other's domain
    m_own = domain_margin(p_other, p_own, psi_own, std.domain_a, std.domain_b)       # other inside own's domain
    inside = m_other <= 1.0 or m_own <= 1.0
    sev = severity(dist, dcpa, tcpa, inside, std)
    speed = math.hypot(v_own[0], v_own[1])
    rel_speed = math.hypot(v_other[0] - v_own[0], v_other[1] - v_own[1])
    return dict(dist=dist, dcpa=dcpa, tcpa=tcpa, severity=sev, inside_own=m_own <= 1.0, inside_other=m_other <= 1.0,
                margin=min(m_other, m_own), cri=collision_risk_index(dcpa, tcpa, speed, std),
                bearing=wrap_pi(math.atan2(p_other[1] - p_own[1], p_other[0] - p_own[0]) - psi_own),
                encounter=encounter_type(psi_own, psi_other, p_own, p_other, rel_speed, speed))


# ------------------------------------------------------------------ grading (v2 grading.py)
def grade_event(ev, std):
    """Event grade (max severity) and the v2 event score 0-100 with the norms at this scale."""
    sev = int(ev.get('max_severity', 0))
    min_dcpa = ev.get('min_dcpa', -1)
    if min_dcpa is None or min_dcpa < 0:
        min_dcpa = std.d_safe
    time_in_domain = (ev.get('steps_sev2', 0) + ev.get('steps_sev3', 0)) * ev.get('dt', 6.0)
    closing = max(ev.get('closing_speed_at_min', 0.0), 0.0)
    score = (40.0 * (1.0 - min(min_dcpa / std.d_safe, 1.0))
             + 25.0 * sev / 3.0
             + 15.0 * float(np.clip(ev.get('max_cri', 0.0), 0, 1))
             + 10.0 * min(time_in_domain / std.time_in_domain_norm, 1.0)
             + 10.0 * min(closing / std.closing_norm, 1.0))
    grade = min(sev, 3)
    return grade, GRADE_NAMES[grade], float(round(score, 2))


# ------------------------------------------------------------------ lifecycle (v3 ship_env_v3.py)
class Lifecycle:
    def __init__(self, std, dt):
        self.std = std
        self.dt = float(dt)
        self.n_clear = max(1, int(math.ceil(std.n_clear_s / self.dt)))
        self.phase = 0
        self.clear_count = 0
        self.prev_d = None
        self.step = 0
        self.t_warning = self.t_cpa = self.t_clear = None

    def start(self, d):
        self.prev_d = float(d)

    def update(self, sev, d, dcpa, tcpa):
        """Advance one decision; returns True once the encounter is clear."""
        self.step += 1
        t = self.step * self.dt
        cpa_behind = (not math.isfinite(tcpa)) or tcpa < 0
        if sev >= 1 and self.phase < 1:
            self.phase = 1
        if sev >= 1 and self.t_warning is None:
            self.t_warning = t
        if cpa_behind and 1 <= self.phase < 2:
            self.phase = 2
            self.t_cpa = t
        opening = d > self.prev_d + 1e-6
        if sev == 0 and d > self.std.r_clear and opening and (cpa_behind or self.phase >= 1 or dcpa >= self.std.d_safe):
            self.clear_count += 1
        else:
            self.clear_count = 0
        if self.clear_count >= self.n_clear and self.phase < 3:
            self.phase = 3
            self.t_clear = t
        self.prev_d = float(d)
        return self.phase == 3


__all__ = ['CollisionStandard', 'SCALES', 'V2_REF', 'SEVERITY_NAMES', 'GRADE_NAMES', 'PHASE_NAMES', 'cpa', 'assess',
           'severity', 'domain_margin', 'collision_risk_index', 'encounter_type', 'grade_event', 'Lifecycle', 'NM']
