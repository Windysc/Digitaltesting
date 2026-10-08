"""
event_labels.py -- marks every event an attacker generated: what kind of event, in which COLREG situation, how
the attacker produced it, and whether the system under test (SUT) could have prevented it
(review/STUDY_PLAN_PEER_REVIEW.md, item 7), 2026-10-08.

Input: the *.events.jsonl files that study_compare.py run writes (one action sequence per encounter; kind
'event' for an induced failure, 'closest' for the closest approach when the method induced none). The
environment is deterministic, so each sequence replays exactly; replay_ok = 0 flags a replay that does not
reproduce the recorded outcome.

Labels (from the SUT's side; bearings positive to port)
  event_type        collision (severity 3 reached) | domain_infringement (held for hold_steps decisions)
  colreg_warning    COLREG situation at the first CPA warning (encounter_standard.encounter_type with the SUT as
                    own ship): head_on | crossing_starboard (attacker on the SUT's starboard side, SUT give-way) |
                    crossing_port (SUT stand-on) | overtaken (attacker overtaking, SUT stand-on) |
                    overtaking (SUT overtaking, SUT give-way) | parallel
  sut_role          give_way | stand_on | both (head-on) | none (parallel), at the first warning
  colreg_event      the situation at the first decision with severity >= 2 (the event decision)
  sector            bearing of the attacker from the SUT's heading at the event decision: ahead | bow_port |
                    bow_stbd | beam_port | beam_stbd | quarter_port | quarter_stbd | astern
                    (boundaries 22.5, 67.5, 112.5, 157.5 deg)
  closing           relative speed at the event decision: fast (>= cruise) | moderate (>= 0.5 cruise) | slow
  sut_state         evading | route: the SUT executing an avoidance alteration at the event decision or not
  generation        how the attacker's commands produced the event, counted up to the event decision:
                      direct       no turn command and no speed change
                      single       one turn segment and at most one speed-change phase
                      multi_phase  two or three turn segments, or turning and speed change in separate phases
                      tracking     four or more turn segments (continuous closed-loop corrections)
  response_lift     how specifically the attacker answers the SUT: share of decisions with a turn toward the SUT
                    within two decisions after a SUT course change (> 5 deg in one decision) minus the same share
                    at all other decisions before the event. A pursuer that always turns toward the SUT gives about
                    0, an open-loop plan (random search, CEM) gives the chance level about 0, a reaction gives > 0.
  counterfactual    the SUT follows its preset until t0 and then one manoeuvre of the library: course change 0,
                    +-30, +-60, +-90 deg from its heading at t0 (+ = port) x speed 0 (stop), 0.5, 1.0 of cruise,
                    within its turn rate and +-0.1 m/s^2; t0 = the first CPA warning or the scenario start.
                    A speed increase is left out: above the attacker's top speed (= cruise) it escapes trivially.
  attribution       open loop (study plan item 7 as written): the attacker's recorded actions replay unchanged
                    (its kinematics do not depend on the SUT), then it holds course and speed; checked until the
                    recorded event time + 60 s.
                      preventable           some manoeuvre started at the first CPA warning avoids the failure
                      not_demonstrated      none started at the warning, some started at the scenario start
                      initially_infeasible  none from either time
                    hold_avoids: holding course and speed from the warning avoids it (the SUT's own alterations
                    took part in the failure); colreg_prevention: an avoiding manoeuvre at the warning with a
                    starboard or no course change exists.
  cl_attribution    closed loop: the attacker's policy reacts to the counterfactual SUT (the PPO policy with argmax
                    actions, the scripted function; an open-loop plan of random search or CEM replays as it is), run
                    to the end of the encounter: escapable (some manoeuvre at the warning avoids the failure) |
                    escapable_from_start | inescapable | not_run (PPO events from sampled attempts).
  adaptive_failure  preventable in open loop and not escapable at the warning in closed loop: the failure comes
                    from the attacker's reaction to whatever the SUT does within the library.
  mark              <colreg_warning>/<sector>/<closing>/<generation>/<attribution>/<cl_attribution>

  python event_labels.py label --src <compare dir> [--src <dir2>] --out <dir> --ppo_root <ppo runs> [--suts replan]
  python event_labels.py summary --out <dir>
"""
import argparse
import glob
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import encounter_standard as S          # noqa: E402
import study_compare as C               # noqa: E402

LIB_COURSE = (0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0)    # deg; + = port (counterclockwise), - = starboard
LIB_SPEED = (0.0, 0.5, 1.0)                                    # x cruise (stop, half, keep)
EXTEND_S = 60.0
ROLE = {'head_on': 'both', 'crossing_starboard': 'give_way', 'overtaking': 'give_way', 'crossing_port': 'stand_on',
        'overtaken': 'stand_on', 'parallel': 'none'}
SECTOR_ORDER = ['ahead', 'bow_port', 'bow_stbd', 'beam_port', 'beam_stbd', 'quarter_port', 'quarter_stbd', 'astern']
GENERATION_ORDER = ['direct', 'single', 'multi_phase', 'tracking']
ATTRIBUTION_ORDER = ['preventable', 'not_demonstrated', 'initially_infeasible']
CL_ORDER = ['escapable', 'escapable_from_start', 'inescapable', 'not_run']


def sector_name(brg_deg):
    a = abs(brg_deg)
    side = 'port' if brg_deg > 0 else 'stbd'
    if a < 22.5:
        return 'ahead'
    if a < 67.5:
        return 'bow_' + side
    if a < 112.5:
        return 'beam_' + side
    if a < 157.5:
        return 'quarter_' + side
    return 'astern'


def sut_view(env):
    """COLREG situation, bearing of the attacker [deg, + port] and relative speed, from the SUT's side."""
    own, tgt = env.live_ownship, env.target
    psi_t, psi_a = math.radians(tgt.psi), math.radians(own.cog)
    vt, va = tgt.velocity, own.velocity
    rel = math.hypot(va[0] - vt[0], va[1] - vt[1])
    enc = S.encounter_type(psi_t, psi_a, (tgt.x, tgt.y), (own.x, own.y), rel, tgt.speed)
    brg = math.degrees(S.wrap_pi(math.atan2(own.y - tgt.y, own.x - tgt.x) - psi_t))
    return enc, brg, rel


def replay(env, ev, t0=None, manoeuvre=None, extend_s=0.0, record=False):
    """Replay the event's actions; with a manoeuvre (course change deg, speed factor) the SUT switches to it at
    the first decision boundary at or after t0. After the recorded actions the attacker holds course and speed for
    extend_s. Returns (evaluation dict, per-decision trace or None)."""
    C.reset_to(env, ev['encounter'])
    acts, zero = ev['actions'], C.zero_action(env)
    t_end = len(acts) * env.dt_decision + extend_s
    trace = [] if record else None
    done, k, switched = False, 0, manoeuvre is None
    while not done and env.t < t_end - 1e-6:
        tgt, own = env.target, env.live_ownship
        if not switched and env.t >= t0 - 1e-6:
            tgt.set_override(tgt.psi + manoeuvre[0], env.cruise * manoeuvre[1])
            switched = True
        a = acts[k] if k < len(acts) else zero
        psi_before = tgt.psi
        if record:
            brg_a = math.degrees(S.wrap_pi(math.atan2(tgt.y - own.y, tgt.x - own.x) - math.radians(own.cog)))
        _, _, done, _ = env.step(a)
        if record:
            enc, brg, rel = sut_view(env)
            trace.append(dict(k=k, t=env.t, acc=env.action_table[a][0], rot=env.action_table[a][1],
                              att_x=own.x, att_y=own.y, att_cog=own.cog, att_sp=own.sp, brg_att_to_sut=brg_a,
                              sut_x=tgt.x, sut_y=tgt.y, sut_psi=tgt.psi, sut_sp=tgt.speed,
                              sut_dpsi=E_wrap(tgt.psi - psi_before), evading=int(getattr(tgt, 'evading', False)),
                              severity=int(env.enc['severity']), dist=env.enc['dist'], colreg=enc, brg=brg, rel=rel))
        k += 1
    return env.evaluation(), trace


def E_wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


def segments(signs):
    """Number of maximal runs of equal non-zero sign."""
    n, prev = 0, 0
    for s in signs:
        if s != 0 and s != prev:
            n += 1
        prev = s
    return n


def generation_labels(trace, k_event, cruise, dt):
    pre = trace[:k_event + 1]
    rot = [int(np.sign(r['rot'])) for r in pre]
    n_turn = segments(rot)
    sp = np.array([cruise] + [r['att_sp'] for r in pre])
    dv = np.diff(sp)
    speed_sign = [int(np.sign(d)) if abs(d) > 0.05 else 0 for d in dv]
    n_speed = segments(speed_sign)
    turning = np.array([r != 0 for r in rot])
    speeding = np.array([s != 0 for s in speed_sign])
    separate = bool(n_speed and n_turn and not np.any(turning & speeding)) if len(pre) else False
    if n_turn == 0 and n_speed == 0:
        gen = 'direct'
    elif n_turn >= 4:
        gen = 'tracking'
    elif n_turn >= 2 or (n_turn == 1 and n_speed >= 1 and separate) or n_speed >= 2:
        gen = 'multi_phase'
    else:
        gen = 'single'
    # reaction to the SUT: turn toward the SUT within two decisions after a SUT course change (> 5 deg in one
    # decision) against the same rate at the other decisions
    changes = [i for i, r in enumerate(pre) if abs(r['sut_dpsi']) > 5.0]
    after = set(j for i in changes for j in (i + 1, i + 2) if j < len(pre))
    toward = [int(r['rot'] != 0 and np.sign(r['rot']) == np.sign(r['brg_att_to_sut'])) for r in pre]
    a_ = [toward[j] for j in range(len(pre)) if j in after]
    b_ = [toward[j] for j in range(len(pre)) if j not in after]
    lift = (np.mean(a_) - np.mean(b_)) if (a_ and b_) else float('nan')
    return dict(generation=gen, turn_segments=n_turn, speed_phases=n_speed, sut_course_changes=len(changes),
                toward_rate=round(float(np.mean(toward)), 3) if toward else -1.0,
                response_lift=round(float(lift), 3) if np.isfinite(lift) else float('nan'),
                min_speed=round(float(sp.min()), 2), max_speed=round(float(sp.max()), 2),
                total_turn_deg=round(float(sum(abs(r['rot']) for r in pre) * dt), 1),
                decisions_to_event=k_event + 1)


def run_counterfactual(env, ev, policy, t0, manoeuvre, t_end=None):
    """The SUT switches to the manoeuvre at the first decision boundary at or after t0; policy(env, obs, k) gives
    the attacker's actions; runs until the episode ends or t_end. Returns the evaluation dict."""
    obs = C.reset_to(env, ev['encounter'])
    done, k, switched = False, 0, False
    while not done and (t_end is None or env.t < t_end - 1e-6):
        if not switched and env.t >= t0 - 1e-6:
            tgt = env.target
            tgt.set_override(tgt.psi + manoeuvre[0], env.cruise * manoeuvre[1])
            switched = True
        obs, _, done, _ = env.step(policy(env, obs, k))
        k += 1
    return env.evaluation()


def library_search(env, ev, policy, t_warning, t_end):
    out = {}
    for name, t0 in (('warning', t_warning), ('start', 0.0)):
        avoid, best = [], (None, -1.0)
        for dc in LIB_COURSE:
            for f in LIB_SPEED:
                res = run_counterfactual(env, ev, policy, t0, (dc, f), t_end)
                if res['outcome'] != 'success':
                    avoid.append((dc, f))
                if res['min_distance'] > best[1]:
                    best = ((dc, f), res['min_distance'])
        out[name] = (avoid, best)
    return out


def open_loop_policy(env, ev):
    acts, zero = ev['actions'], C.zero_action(env)
    return lambda e, o, k: acts[k] if k < len(acts) else zero


def attribution(env, ev, t_warning):
    """Open loop, as written in study plan item 7."""
    out = library_search(env, ev, open_loop_policy(env, ev), t_warning, len(ev['actions']) * env.dt_decision + EXTEND_S)
    aw, bw = out['warning']
    label = 'preventable' if aw else ('not_demonstrated' if out['start'][0] else 'initially_infeasible')
    return dict(attribution=label, n_avoid_warning=len(aw), n_avoid_start=len(out['start'][0]),
                hold_avoids=int((0.0, 1.0) in aw), colreg_prevention=int(any(dc <= 0.0 for dc, f in aw)),
                best_at_warning='%+.0f deg x %.2f' % bw[0], best_min_distance=round(bw[1], 1),
                avoiding_at_warning=';'.join('%+.0f/%.2f' % m for m in aw))


def closed_loop_attribution(env, ev, policy, t_warning):
    """The attacker's policy reacts to the counterfactual SUT; to the end of the encounter."""
    out = library_search(env, ev, policy, t_warning, None)
    aw, bw = out['warning']
    label = 'escapable' if aw else ('escapable_from_start' if out['start'][0] else 'inescapable')
    return dict(cl_attribution=label, cl_n_avoid_warning=len(aw), cl_n_avoid_start=len(out['start'][0]),
                cl_hold_avoids=int((0.0, 1.0) in aw), cl_colreg_escape=int(any(dc <= 0.0 for dc, f in aw)),
                cl_best_at_warning='%+.0f deg x %.2f' % bw[0], cl_best_min_distance=round(bw[1], 1),
                cl_avoiding_at_warning=';'.join('%+.0f/%.2f' % m for m in aw))


def attacker_policy(ev, env, ppo_root, cache):
    """Closed-loop policy that generated the event, or None when it is stochastic (sampled PPO attempts)."""
    m = ev['method']
    if m in C.BASELINES:
        fn = C.BASELINES[m]
        return lambda e, o, k: fn(e, o)
    if m.startswith('ppo'):
        if ev['attempt'] != 1 or not ppo_root:
            return None
        key = (ev['sut'], m, ev['seed'])
        if key not in cache:
            agent, _ = C.load_ppo(os.path.join(ppo_root, '%s_%s_s%d' % (ev['sut'], m.split('_', 1)[1], ev['seed'])), env)
            cache[key] = agent
        agent = cache[key]
        return lambda e, o, k: agent.select_action(o, deterministic=True)
    return open_loop_policy(env, ev)                 # random search, CEM: the plan is the policy


def label_file(path, with_attribution=True, ppo_root=''):
    rows = []
    evs = [json.loads(l) for l in open(path) if l.strip()]
    if not evs:
        return rows
    e0 = evs[0]
    encs = C.encounter_list(e0.get('scenario_spec', 'mix'), e0.get('per_scenario', 6), e0.get('enc_seed', 20000))
    env = C.make_env(e0['sut'], encs, v_max=e0.get('v_max', 6.0), split=e0.get('split', 'test'),
                     save_dir=os.path.dirname(path))
    cache = {}
    for ev in evs:
        res, tr = replay(env, ev, record=True)
        t_warning = env.lifecycle.t_warning
        is_event = ev['kind'] == 'event'
        replay_ok = int((res['outcome'] == 'success') == is_event and res['steps'] == len(ev['actions']))
        sev = [r['severity'] for r in tr]
        k_ev = next((i for i, s in enumerate(sev) if s >= 2), int(np.argmin([r['dist'] for r in tr])))
        k_w = next((i for i, s in enumerate(sev) if s >= 1), k_ev)
        r_ev, r_w = tr[k_ev], tr[k_w]
        row = dict(sut=ev['sut'], method=ev['method'], seed=ev['seed'], encounter=ev['encounter'],
                   scenario=ev['scenario'], family=res['family'], kind=ev['kind'], attempt=ev['attempt'],
                   replay_ok=replay_ok, outcome=res['outcome'],
                   event_type=('collision' if res['max_severity'] >= 3 else 'domain_infringement') if is_event else 'none',
                   min_distance=res['min_distance'], t_warning=t_warning if t_warning is not None else -1.0,
                   t_event=round(r_ev['t'], 1), colreg_warning=r_w['colreg'], sut_role=ROLE.get(r_w['colreg'], 'none'),
                   colreg_event=r_ev['colreg'], sector=sector_name(r_ev['brg']), bearing_event=round(r_ev['brg'], 1),
                   rel_speed_event=round(r_ev['rel'], 2),
                   closing='fast' if r_ev['rel'] >= env.cruise else ('moderate' if r_ev['rel'] >= 0.5 * env.cruise else 'slow'),
                   sut_state='evading' if r_ev['evading'] else 'route', sut_alterations=res['target_evasions'],
                   sut_replans=getattr(env.target, 'n_replans', 0))
        row.update(generation_labels(tr, k_ev, env.cruise, env.dt_decision))
        t0 = t_warning if t_warning is not None else r_ev['t']
        if is_event and with_attribution:
            row.update(attribution(env, ev, t0))
            pol = attacker_policy(ev, env, ppo_root, cache)
            if pol is not None:
                row.update(closed_loop_attribution(env, ev, pol, t0))
                row['adaptive_failure'] = int(row['attribution'] == 'preventable' and row['cl_attribution'] != 'escapable')
            else:
                row.update(cl_attribution='not_run')
        else:
            row.update(attribution='no_event' if not is_event else 'not_run', cl_attribution='no_event' if not is_event else 'not_run')
        row['mark'] = '/'.join([row['colreg_warning'], row['sector'], row['closing'], row['generation'], row['attribution'],
                                row['cl_attribution']])
        rows.append(row)
    return rows


def label(args):
    import pandas as pd
    files = []
    for src in args.src:
        files += glob.glob(os.path.join(src, '*', '*.events.jsonl'))
    suts = set(args.suts.split(',')) if args.suts else None
    methods = set(args.methods.split(',')) if args.methods else None

    def keep(f):
        sut = os.path.basename(os.path.dirname(f))
        meth = os.path.basename(f).rsplit('_s', 1)[0]
        return (suts is None or sut in suts) and (methods is None or meth in methods)
    files = sorted(f for f in files if keep(f))
    print('%d event files' % len(files), flush=True)
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for f, part in zip(files, ex.map(label_file, files, [not args.no_attribution] * len(files),
                                         [args.ppo_root] * len(files))):
            for r in part:
                r['source'] = os.path.basename(os.path.dirname(os.path.dirname(f)))
            rows += part
            print('%-48s %3d entries' % (os.path.relpath(f, os.path.dirname(os.path.dirname(os.path.dirname(f)))),
                                         len(part)), flush=True)
    os.makedirs(args.out, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, 'event_labels.csv'), index=False)
    print('wrote %d rows, replay_ok %.3f' % (len(df), df['replay_ok'].mean()))


def entropy(counts):
    p = np.asarray(counts, float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0


def summary(args):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    df = pd.read_csv(os.path.join(args.out, 'event_labels.csv'))
    ev = df[df['kind'] == 'event'].copy()
    tabs = []
    for col in ('event_type', 'colreg_warning', 'sut_role', 'sector', 'closing', 'sut_state', 'generation', 'attribution',
                'cl_attribution'):
        t = pd.crosstab([ev['sut'], ev['method']], ev[col], normalize='index').round(3)
        t.columns = ['%s=%s' % (col, c) for c in t.columns]
        tabs.append(t)
    shares = pd.concat(tabs, axis=1)
    g = ev.groupby(['sut', 'method'])
    base = g.agg(events=('mark', 'size'), seeds=('seed', 'nunique'), replay_ok=('replay_ok', 'mean'),
                 toward_rate=('toward_rate', 'mean'), response_lift=('response_lift', 'mean'),
                 hold_avoids=('hold_avoids', 'mean'), colreg_prevention=('colreg_prevention', 'mean'),
                 distinct_marks=('mark', 'nunique'))
    cl = ev[ev['cl_attribution'].isin(CL_ORDER[:3])].groupby(['sut', 'method'])
    rr = pd.concat([cl.size().rename('cl_events'), cl['adaptive_failure'].mean().rename('adaptive_failure'),
                    cl['cl_colreg_escape'].mean().rename('cl_colreg_escape')], axis=1)
    # diversity per seed: distinct situation x sector x generation classes among one seed's events, normalised entropy
    ev['mech'] = ev['colreg_warning'] + '/' + ev['sector'] + '/' + ev['generation']
    div = ev.groupby(['sut', 'method', 'seed'])['mech'].agg(lambda s: (s.nunique(), entropy(s.value_counts().values)))
    div = pd.DataFrame(div.tolist(), index=div.index, columns=['classes_per_seed', 'entropy_per_seed']).groupby(['sut', 'method']).mean().round(3)
    S_ = pd.concat([base, rr, div, shares], axis=1).reset_index()
    S_.to_csv(os.path.join(args.out, 'label_summary.csv'), index=False)
    marks = ev.groupby(['sut', 'method', 'mark']).size().rename('n').reset_index().sort_values(['sut', 'method', 'n'], ascending=[True, True, False])
    marks.to_csv(os.path.join(args.out, 'marks_by_method.csv'), index=False)
    pd.set_option('display.width', 250)
    print(S_[['sut', 'method', 'events', 'seeds', 'replay_ok', 'distinct_marks', 'classes_per_seed', 'entropy_per_seed',
              'toward_rate', 'response_lift', 'hold_avoids', 'colreg_prevention', 'cl_events', 'adaptive_failure',
              'cl_colreg_escape']].to_string(index=False))
    for sut in sorted(ev['sut'].unique()):
        e = ev[ev['sut'] == sut]
        meths = [m for m in METHOD_ORDER if m in set(e['method'])] + sorted(set(e['method']) - set(METHOD_ORDER))
        fig, axs = plt.subplots(1, 5, figsize=(22, 0.55 * len(meths) + 1.8))
        for ax, (col, order, title) in zip(axs, (('attribution', ATTRIBUTION_ORDER, 'open-loop attribution (attacker path fixed)'),
                                                 ('cl_attribution', CL_ORDER, 'closed-loop attribution (attacker reacts)'),
                                                 ('colreg_warning', None, 'COLREG situation at the first warning (SUT side)'),
                                                 ('sector', SECTOR_ORDER, 'attacker sector at the event (SUT heading)'),
                                                 ('generation', GENERATION_ORDER, 'generation mode of the attacker'))):
            t = pd.crosstab(e['method'], e[col], normalize='index').reindex(meths)
            cols = [c for c in (order or sorted(t.columns)) if c in t.columns]
            left = np.zeros(len(t))
            cmap = plt.get_cmap('tab10' if len(cols) <= 10 else 'tab20')
            for j, c in enumerate(cols):
                ax.barh(range(len(t)), t[c].values, left=left, color=cmap(j), label=DISPLAY.get(c, c))
                left += t[c].fillna(0).values
            ax.set_yticks(range(len(t)))
            ax.set_yticklabels(['%s (n=%d)' % (DISPLAY.get(m, m), (e['method'] == m).sum()) for m in t.index]
                               if ax is axs[0] else [])
            ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_title(title, fontsize=9); ax.set_xlabel('share of events')
            ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(0.0, -0.2), ncol=1, frameon=False)
        fig.suptitle('Event marks, SUT: %s' % sut, fontsize=11)
        fig.tight_layout(); fig.savefig(os.path.join(args.out, 'marks_%s.png' % sut), dpi=130, bbox_inches='tight')
        plt.close(fig)
    if args.catalogue:
        catalogue(args, ev)


METHOD_ORDER = ['intercept', 'pursuit', 'hold', 'random', 'cem', 'random_b300', 'cem_b300', 'ppo_mc', 'ppo_gae']
DISPLAY = {
    # methods
    'intercept': 'intercept (scripted)', 'pursuit': 'pursuit (scripted)', 'hold': 'hold course', 'random': 'random search',
    'cem': 'CEM', 'ppo_mc': 'PPO (Monte-Carlo update)', 'ppo_gae': 'PPO (GAE)',
    # attribution
    'preventable': 'preventable at the warning', 'not_demonstrated': 'not demonstrated (only from the start)',
    'initially_infeasible': 'initially infeasible', 'escapable': 'escapable at the warning',
    'escapable_from_start': 'escapable from the start only', 'inescapable': 'inescapable',
    'not_run': 'not run (sampled PPO attempt)',
    # COLREG situation, SUT side
    'crossing_port': 'crossing, attacker to port (SUT stand-on)',
    'crossing_starboard': 'crossing, attacker to starboard (SUT give-way)', 'head_on': 'head-on (both give way)',
    'overtaken': 'overtaken by the attacker (SUT stand-on)', 'overtaking': 'SUT overtaking (SUT give-way)',
    'parallel': 'parallel lanes',
    # sectors
    'ahead': 'ahead', 'bow_port': 'port bow', 'bow_stbd': 'starboard bow', 'beam_port': 'port beam',
    'beam_stbd': 'starboard beam', 'quarter_port': 'port quarter', 'quarter_stbd': 'starboard quarter', 'astern': 'astern',
    # generation
    'direct': 'direct (no manoeuvre)', 'single': 'single manoeuvre', 'multi_phase': 'multi-phase manoeuvre',
    'tracking': 'tracking (closed-loop corrections)'}


def catalogue(args, ev):
    """One example per frequent mechanism (situation at the warning / sector at the event / generation mode) of one
    method against one SUT: both tracks, the event point, the SUT's domain; the title gives the attribution split."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    sut, meth = args.catalogue.split(':')
    e = ev[(ev['sut'] == sut) & (ev['method'] == meth)]
    top = e['mech'].value_counts().head(args.n_examples)
    src = {}
    for path in glob.glob(os.path.join(args.events_root, sut, '%s_s*.events.jsonl' % meth)):
        for l in open(path):
            d = json.loads(l)
            src[(d['seed'], d['encounter'])] = d
    n = len(top)
    cols = min(4, n)
    rows_ = int(math.ceil(n / cols))
    fig, axs = plt.subplots(rows_, cols, figsize=(5.0 * cols, 5.0 * rows_), squeeze=False)
    env = None
    for ax, (mark, count) in zip(axs.ravel(), top.items()):
        sel = e[e['mech'] == mark]
        r = sel.iloc[0]
        split = ', '.join('%d %s' % (v, DISPLAY.get(k, k)) for k, v in sel['attribution'].value_counts().items())
        split_cl = ', '.join('%d %s' % (v, DISPLAY.get(k, k)) for k, v in sel['cl_attribution'].value_counts().items())
        d = src[(r['seed'], r['encounter'])]
        if env is None:
            encs = C.encounter_list(d.get('scenario_spec', 'mix'), d.get('per_scenario', 6), d.get('enc_seed', 20000))
            env = C.make_env(sut, encs, v_max=d.get('v_max', 6.0), split=d.get('split', 'test'), save_dir=args.out)
        _, tr = replay(env, d, record=True)
        ax_ = np.array([[t['att_x'], t['att_y']] for t in tr]); st = np.array([[t['sut_x'], t['sut_y']] for t in tr])
        k = next((i for i, t in enumerate(tr) if t['severity'] >= 2), len(tr) - 1)
        ax.plot(st[:, 0], st[:, 1], '-', color='tab:red', lw=1.4, label='SUT (%s)' % sut)
        ax.plot(ax_[:, 0], ax_[:, 1], '-', color='tab:blue', lw=1.4, label='attacker (%s)' % meth)
        ax.plot(*st[0], 'o', color='tab:red', ms=4); ax.plot(*ax_[0], 'o', color='tab:blue', ms=4)
        ax.plot(*ax_[k], 'x', color='black', ms=8, mew=2)
        t_ = tr[k]
        ax.add_patch(Ellipse((t_['sut_x'], t_['sut_y']), 2 * env.std.domain_a, 2 * env.std.domain_b, angle=t_['sut_psi'],
                             fc='none', ec='tab:red', ls='--'))
        lo = np.minimum(ax_.min(0), st.min(0)) - 150; hi = np.maximum(ax_.max(0), st.max(0)) + 150
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect('equal'); ax.grid(alpha=.3)
        import textwrap
        lines = [' / '.join(DISPLAY.get(p, p) for p in mark.split('/')),
                 '%d of %d events (example: %s, seed %d, encounter %d)' % (count, len(e), r['scenario'], r['seed'],
                                                                           r['encounter']),
                 'open loop: ' + split, 'closed loop: ' + split_cl]
        ax.set_title('\n'.join(textwrap.fill(l, 62) for l in lines), fontsize=6.8)
        ax.tick_params(labelsize=7)
        ax.set_xlabel('east [m]', fontsize=7); ax.set_ylabel('north [m]', fontsize=7)
    for ax in axs.ravel()[n:]:
        ax.axis('off')
    axs[0][0].legend(fontsize=7, loc='best')
    fig.suptitle('Most frequent event mechanisms of %s against %s (situation at the first warning / attacker sector at '
                 'the event / generation mode; x: first decision inside the ship domain; dashed: SUT domain at that moment)'
                 % (DISPLAY.get(meth, meth), sut), fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, 'catalogue_%s_%s.png' % (sut, meth)), dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['label', 'summary'])
    ap.add_argument('--src', action='append', default=[], help='compare folder(s) holding <sut>/*.events.jsonl')
    ap.add_argument('--out', required=True)
    ap.add_argument('--suts', default='')
    ap.add_argument('--methods', default='')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--no_attribution', action='store_true')
    ap.add_argument('--ppo_root', default='', help='label: PPO run folders <sut>_<update>_s<seed>, for the closed loop')
    ap.add_argument('--catalogue', default='', help='summary: <sut>:<method> for the example figure')
    ap.add_argument('--events_root', default='', help='summary: compare folder of the catalogue method')
    ap.add_argument('--n_examples', type=int, default=8)
    args = ap.parse_args()
    {'label': label, 'summary': summary}[args.cmd](args)


if __name__ == '__main__':
    main()
