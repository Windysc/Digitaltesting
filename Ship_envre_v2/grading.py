"""
Grading of the events an episode produces, and scoring of an agent.

Event grade (maritime terms, from the episode's encounter history):

  grade 0  safe passage            no CPA warning at any time
  grade 1  close-quarters situation CPA warning (DCPA < d_safe within t_safe)
                                   but no domain infringement
  grade 2  domain infringement     the ship domain of either ship was
           (near miss)             violated, no collision
  grade 3  collision               centre distance below the collision distance

Event score 0-100 (continuous, higher = more dangerous event):
  40 * (1 - min(min_DCPA / d_safe, 1))          how far inside the safe distance
  25 * max_severity / 3                          the ladder level reached
  15 * max_CRI                                   collision-risk index peak
  10 * min(time_in_domain / 300 s, 1)            how long the domain was held
  10 * min(closing speed at min range / 10 m/s, 1)  how hard the approach was

Attack-agent ability on a set of episodes:
  success_rate   fraction with the required severity held
  mean_score     mean event score
  efficiency     mean (baseline steps / agent steps) on successful episodes,
                 1 = as fast as the scripted baseline
  effort         mean rudder-level change per step (lower = more ship-like)
  robustness     min success rate over the automation levels tried
Navigation-agent ability uses the same fields with the sign of the event
score reversed (safe passages score high) and a COLREG-compliance rate.
"""
import numpy as np

GRADE_NAMES = {0: 'safe_passage', 1: 'close_quarters', 2: 'domain_infringement', 3: 'collision'}


def grade_event(ev, params):
    """ev: the environment's evaluation() dict; params: EncounterParams."""
    sev = int(ev.get('max_severity', 0))
    grade = min(sev, 3)
    min_dcpa = ev.get('min_dcpa', -1)
    if min_dcpa is None or min_dcpa < 0:
        min_dcpa = params.d_safe
    time_in_domain = (ev.get('steps_sev2', 0) + ev.get('steps_sev3', 0)) * ev.get('dt', 10.0)
    closing = max(ev.get('closing_speed_at_min', 0.0), 0.0)
    score = (40.0 * (1.0 - min(min_dcpa / params.d_safe, 1.0))
             + 25.0 * sev / 3.0
             + 15.0 * float(np.clip(ev.get('max_cri', 0.0), 0, 1))
             + 10.0 * min(time_in_domain / 300.0, 1.0)
             + 10.0 * min(closing / 10.0, 1.0))
    return grade, GRADE_NAMES[grade], float(round(score, 2))


def colreg_compliance(ev):
    """Own-ship compliance at the first CPA warning (navigation task):
    give-way ships must alter to starboard (course change < 0 in the math
    convention), stand-on ships must hold course; unknown roles pass."""
    role = ev.get('colreg_role', 'unknown')
    dpsi = ev.get('course_change_after_warning_deg', None)
    if dpsi is None or role == 'unknown':
        return None
    if role in ('give_way', 'both_give_way'):
        return bool(dpsi <= -10.0)
    if role == 'stand_on':
        return bool(abs(dpsi) <= 15.0)
    return None


def summarise(records, task, baseline_steps=None):
    """Aggregate a list of evaluation dicts (each already carrying grade/score)."""
    if not records:
        return {}
    succ = np.array([r['success'] for r in records], float)
    score = np.array([r['event_score'] for r in records], float)
    grades = np.array([r['event_grade'] for r in records], int)
    out = dict(n=len(records), success_rate=float(succ.mean()), mean_score=float(score.mean()),
               grade_hist={GRADE_NAMES[g]: int((grades == g).sum()) for g in range(4)},
               effort=float(np.mean([r['rudder_change_mean'] for r in records])),
               mean_steps=float(np.mean([r['steps'] for r in records])))
    ok = [r for r in records if r['success']]
    if ok:
        steps = np.array([r['destination_step'] if 'destination_step' in r else r['capture_step'] for r in ok], float)
        steps = np.where(steps > 0, steps, np.array([r['steps'] for r in ok], float))
        out['mean_steps_success'] = float(steps.mean())
        if baseline_steps:
            out['efficiency'] = float(np.mean(baseline_steps / steps))
    if task == 'navigate':
        comp = [colreg_compliance(r) for r in records]
        comp = [c for c in comp if c is not None]
        out['colreg_compliance'] = float(np.mean(comp)) if comp else None
        out['safety_score'] = float(100.0 - score.mean())
    return out
