"""
Maritime encounter assessment: CPA, ship domain, collision-risk index and
COLREG encounter classification.  Replaces the "distance < 300 m" capture
rule of the original environment with the criteria used in navigation
safety analysis.

Conventions: x east, y north, headings in radians counter-clockwise from +x
(math convention, as in the rest of the package).  A positive relative
bearing is on the port side.

Default parameters (all configurable through EncounterParams):
  d_safe   926 m  (0.5 nautical mile)   DCPA below which an approach is unsafe
  t_safe   900 s  (15 min)              TCPA window in which that DCPA matters
  domain   ellipse semi-axes 4 L along the course and 1.6 L across it,
           i.e. the 8 L x 3.2 L ellipse of Fujii & Tanaka (1971); a
           circular alternative is domain_b_L = domain_a_L
  collision  distance below 1.0 L between centres (hull contact for two
           ships of this size)
  CRI      cr_al ** ((DCPA + v * TCPA) / dr) with dr = 2000 m and
           cr_al = 0.2, the formula that Ship_envre/Ship_env.py.cr_cal
           carried (unused there); equals 1 at DCPA + v*TCPA = 0 and 0.2 at
           2000 m.

Severity levels used by the environment:
  0 clear
  1 CPA warning      DCPA < d_safe and 0 <= TCPA < t_safe
  2 domain violation the other ship is inside this ship's domain ellipse
  3 collision        centre distance < collision_L * L
"""
import numpy as np


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class EncounterParams:
    def __init__(self, d_safe=926.0, t_safe=900.0, domain_a_L=4.0, domain_b_L=1.6,
                 collision_L=1.0, cri_dr=2000.0, cri_al=0.2, L=244.74):
        self.d_safe = float(d_safe)
        self.t_safe = float(t_safe)
        self.domain_a = float(domain_a_L) * L
        self.domain_b = float(domain_b_L) * L
        self.collision_dist = float(collision_L) * L
        self.cri_dr = float(cri_dr)
        self.cri_al = float(cri_al)
        self.L = L

    def as_dict(self):
        return dict(d_safe=self.d_safe, t_safe=self.t_safe, domain_a=self.domain_a, domain_b=self.domain_b,
                    collision_dist=self.collision_dist, cri_dr=self.cri_dr, cri_al=self.cri_al)


def cpa(p_own, v_own, p_other, v_other):
    """DCPA [m] and TCPA [s] of two ships with constant velocities.
    TCPA < 0 means the closest point of approach has already passed; DCPA is
    then the current distance."""
    r = np.asarray(p_other, float) - np.asarray(p_own, float)
    vr = np.asarray(v_other, float) - np.asarray(v_own, float)
    vr2 = float(vr @ vr)
    dist = float(np.hypot(r[0], r[1]))
    if vr2 < 1e-9:
        return dist, np.inf
    tcpa = -float(r @ vr) / vr2
    if tcpa < 0:
        return dist, tcpa
    dcpa = float(np.linalg.norm(r + vr * tcpa))
    return dcpa, tcpa


def in_domain(p_point, p_ship, psi_ship, a, b, forward_shift=0.0):
    """True if p_point lies inside the ellipse (semi-axes a along the ship's
    course, b across) centred `forward_shift` metres ahead of p_ship."""
    d = np.asarray(p_point, float) - np.asarray(p_ship, float)
    c, s = np.cos(psi_ship), np.sin(psi_ship)
    xb = c * d[0] + s * d[1] - forward_shift
    yb = -s * d[0] + c * d[1]
    return (xb / a) ** 2 + (yb / b) ** 2 <= 1.0


def domain_margin(p_point, p_ship, psi_ship, a, b):
    """Normalised ellipse radius of p_point in the ship's domain frame
    (< 1 inside, 1 on the boundary)."""
    d = np.asarray(p_point, float) - np.asarray(p_ship, float)
    c, s = np.cos(psi_ship), np.sin(psi_ship)
    xb = c * d[0] + s * d[1]
    yb = -s * d[0] + c * d[1]
    return float(np.sqrt((xb / a) ** 2 + (yb / b) ** 2))


def collision_risk_index(dcpa, tcpa, speed, params):
    """CRI in [0, 1] from Ship_envre/Ship_env.py.cr_cal (0 once the CPA has passed)."""
    if not np.isfinite(tcpa) or tcpa < 0:
        return 0.0
    x = (dcpa + speed * tcpa) / params.cri_dr
    return float(np.clip(params.cri_al ** x, 0.0, 1.0))


def severity(dist, dcpa, tcpa, inside_domain, params):
    if dist < params.collision_dist:
        return 3
    if inside_domain:
        return 2
    if dcpa < params.d_safe and 0.0 <= tcpa < params.t_safe:
        return 1
    return 0


SEVERITY_NAMES = {0: 'clear', 1: 'cpa_warning', 2: 'domain_violation', 3: 'collision'}


def encounter_type(psi_own, psi_other, p_own, p_other):
    """COLREG situation seen from `own`: head-on (Rule 14), overtaking (Rule
    13, other approaches from > 112.5 deg abaft the beam), crossing from
    starboard / port (Rule 15)."""
    d = np.asarray(p_other, float) - np.asarray(p_own, float)
    brg_from_own = wrap_pi(np.arctan2(d[1], d[0]) - psi_own)          # where the other is, seen from own
    brg_from_other = wrap_pi(np.arctan2(-d[1], -d[0]) - psi_other)    # where own is, seen from the other
    course_diff = abs(wrap_pi(psi_other - psi_own))
    if course_diff > np.radians(174) and abs(brg_from_own) < np.radians(6):
        return 'head_on'
    if abs(brg_from_other) > np.radians(112.5):
        return 'overtaking'
    if abs(brg_from_own) > np.radians(112.5):
        return 'overtaken'
    return 'crossing_starboard' if brg_from_own < 0 else 'crossing_port'


def assess(p_own, v_own, psi_own, p_other, v_other, psi_other, params):
    """Full assessment from `own`'s point of view. Returns a dict."""
    dist = float(np.linalg.norm(np.asarray(p_other) - np.asarray(p_own)))
    dcpa, tcpa = cpa(p_own, v_own, p_other, v_other)
    inside_other = in_domain(p_own, p_other, psi_other, params.domain_a, params.domain_b)
    inside_own = in_domain(p_other, p_own, psi_own, params.domain_a, params.domain_b)
    sev = severity(dist, dcpa, tcpa, inside_other or inside_own, params)
    speed = float(np.linalg.norm(v_own))
    return dict(dist=dist, dcpa=dcpa, tcpa=tcpa, inside_other=bool(inside_other), inside_own=bool(inside_own),
                severity=sev, cri=collision_risk_index(dcpa, tcpa, speed, params),
                margin=domain_margin(p_own, p_other, psi_other, params.domain_a, params.domain_b),
                encounter=encounter_type(psi_own, psi_other, p_own, p_other))
