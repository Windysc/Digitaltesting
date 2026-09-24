"""
check_env_wiring.py -- checks of step 4 of the mending plan for part 1: the route chain of
ais_prep.py wired into the scenario environments (scenario_targets.py, attack_scenarios.py).

    python check_env_wiring.py            # writes check_out/results_env_wiring.md

Uses the mended windows of the synthetic development fleet (check_out/prep_raw_off, made by
check_mending_plan.py) as the trace set.  Every check states what it compares and passes or fails
on a stated bound; the script exits with 1 when any check fails.

Scope: the two fixed-track environments of the pipeline (EncounterAttackEnv, ScenarioAttackEnv).  The
earlier Ship_envre_v2 / v3 builds, whose targets steered on a polyline route by pure pursuit and kept
their own route_from_trace, were removed from the repository on 2026-09-24 and are not covered.
"""
import datetime
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))     # the general folder: ais_prep.py and the environments side by side
sys.path.insert(0, HERE)
import ais_prep as A                    # noqa: E402
import env_moving_obj as E              # noqa: E402
import encounter_standard as S          # noqa: E402
import scenario_targets as ST           # noqa: E402
import attack_scenarios as AS           # noqa: E402

OUT = os.path.join(HERE, 'check_out')
TRACE = os.path.join(OUT, 'prep_raw_off', 'windows_lonlat.npy')
WXY = os.path.join(OUT, 'prep_raw_off', 'windows_xy.npy')
LINES, FAILS = [], []


def say(*a):
    line = ' '.join(str(x) for x in a)
    print(line)
    LINES.append(line)


def table(header, rows):
    say('| ' + ' | '.join(header) + ' |')
    say('|' + '---|' * len(header))
    for r in rows:
        say('| ' + ' | '.join(str(x) for x in r) + ' |')
    say('')


def check(name, ok, detail):
    say('- %s: **%s** (%s)' % (name, 'pass' if ok else 'FAIL', detail))
    if not ok:
        FAILS.append(name)


def q3(a, fmt='%.2f'):
    a = np.asarray(a, float)
    return ' / '.join(fmt % v for v in (np.median(a), np.percentile(a, 95), a.max())) if len(a) else '-'


def track_yaw(track, dt):
    """Yaw rate [deg/s] between consecutive track points (chord headings)."""
    d = np.diff(np.asarray(track, float), axis=0)
    psi = np.degrees(np.unwrap(np.arctan2(d[:, 1], d[:, 0])))
    return np.abs(np.diff(psi)) / dt


def run_episode(env, policy, max_steps):
    env.reset()
    done, n = False, 0
    while not done and n < max_steps:
        _, _, done, _ = env.step(policy(env))
        n += 1
    return n, env.evaluation(), np.asarray(env.target.track)


def w1_loader(shapes, wins):
    say('## W1 Loader: one tangent plane per trace, no smoothing')
    say('')
    # The loader's plane has its north at the window's first point, windows_xy.npy the voyage's; the two differ
    # by the meridian convergence between the origins (a rotation of about 0.17 deg per 12 km of easting at 57.6 deg
    # latitude), which the route chain does not see because a route is placed by its own start tangent.  So the
    # shapes are compared after the best-fit rotation about the first point, and that rotation is reported.
    raw, aligned, rot = [], [], []
    for sh, w in zip(shapes, wins):
        a, b = sh - sh[0], w - w[0]
        raw.append(np.abs(a - b).max())
        ang = math.atan2(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]), np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]))
        R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
        aligned.append(np.abs(a @ R.T - b).max())
        rot.append(abs(math.degrees(ang)))
    say('Loaded trace (tangent plane at its first point) against windows_xy.npy (the voyage plane, shifted to the window '
        'start), %d windows: as loaded %s m, after the best-fit rotation about the first point %s m; that rotation is %s deg '
        '(median / P95 / max).' % (len(raw), q3(raw, '%.1f'), q3(aligned, '%.3f'), q3(rot, '%.3f')))
    say('')
    check('W1 loader shape within 1 m of windows_xy after alignment', max(aligned) < 1.0, 'max %.3f m' % max(aligned))
    check('W1 frame rotation below 0.5 deg (meridian convergence)', max(rot) < 0.5, 'max %.3f deg' % max(rot))
    say('')


def w2_routes(shapes):
    say('## W2 prepare_trace_routes: acceptance under the standard at both scales')
    say('')
    rows, out = [], {}
    for scale_name in ('arena', 'maritime'):
        std = S.CollisionStandard(scale_name, 6.0)
        routes, info = ST.prepare_trace_routes(shapes, 6.0, std.turn_rate_deg, scale=std.scale)
        yaw = [r['fit']['max_yaw_deg_s'] for r in routes]
        rows.append([scale_name, '%.3f' % std.scale, '%.3f' % std.turn_rate_deg, '%d / %d' % (info['n_accepted'], info['n_traces']),
                     '%.0f %%' % (100 * info['share_smoothed']), '%.0f' % info['smooth_m_max'], q3(yaw, '%.3f')])
        out[scale_name] = (std, routes, info)
        check('W2 %s: every accepted route within the limit' % scale_name, max(yaw) <= std.turn_rate_deg * 1.001,
              'largest %.3f vs limit %.3f deg/s' % (max(yaw), std.turn_rate_deg))
    table(['scale', 'length factor', 'yaw limit [deg/s]', 'accepted / traces', 'share needing smoothing', 'largest smoothing [m]',
           'speed x curvature of accepted routes [deg/s] P50 / P95 / max'], rows)
    return out


def played_yaw(target, dt_sub):
    """Yaw rate [deg/s] from the heading the target reported at every sub-step (TargetShip.psi_track)."""
    psi = np.degrees(np.unwrap(np.radians(np.asarray(target.psi_track, float))))
    return np.abs(np.diff(psi)) / dt_sub


def w3_encounter_env(prepared, seeds=(3, 11, 19)):
    say('## W3 EncounterAttackEnv with the trace set: played track against the standard')
    say('')
    say('All scenarios of the mix, %d seeds each, hold-course and intercept attackers, 200 decisions per episode. '
        'The yaw rate is taken from the heading the target reported at every sub-step (TargetShip.psi_track).' % len(seeds))
    say('')
    rows = []
    for scale_name, (std, routes, info) in prepared.items():
        env = AS.EncounterAttackEnv(E.ownship(0, 0, 0, 0, 0, 0), scenario='mix', scale=scale_name, trace=TRACE,
                                    cycle_scenarios=True, seed=3, save_dir=os.path.join(OUT, 'env_tmp'))
        assert env.trace_info['n_accepted'] == info['n_accepted']
        dt_sub = float(env.dt_sub)
        yaw_max, chord_max, head_err, perp, shaped, smooth, outcomes = [], [], [], [], [], [], []
        tracks = {}
        for seed in seeds:
            for policy_name, policy in (('hold', AS.hold_action), ('intercept', AS.intercept_action)):
                env.seed(seed)
                env._cycle_i = 0
                for k in range(len(env.names)):
                    n, ev, track = run_episode(env, policy, 200)
                    tracks.setdefault((seed, k), {})[policy_name] = track
                    if policy_name == 'hold':
                        yaw_max.append(played_yaw(env.target, dt_sub).max())
                        chord_max.append(track_yaw(track, dt_sub).max())
                        # the scenario course in the metadata is rounded to 0.1 deg
                        head_err.append(abs(E.wrap_deg(env.current['course_diff_deg'] - math.degrees(env.target.route_table['psi'][0]))))
                        perp.append(A.perpendicular_distance(track[::10], env.target.route).max())
                        shaped.append(ev['data_shaped'])
                        smooth.append(ev['route_smooth_m'])
                        outcomes.append(ev['outcome'])
        same = []
        for k, tr in tracks.items():
            m = min(len(tr['hold']), len(tr['intercept']))
            same.append(np.abs(tr['hold'][:m] - tr['intercept'][:m]).max())
        rows.append([scale_name, len(yaw_max), q3(yaw_max, '%.3f'), q3(chord_max, '%.3f'), '%.3f' % env.turn_rate,
                     '%.3f' % max(head_err), q3(perp, '%.2f'), '%.1e' % max(same), ', '.join(sorted(set(outcomes)))])
        check('W3 %s: played yaw rate within the limit' % scale_name, max(yaw_max) <= env.turn_rate * 1.001,
              'largest %.3f vs limit %.3f deg/s over %d episodes' % (max(yaw_max), env.turn_rate, len(yaw_max)))
        check('W3 %s: first heading equals the scenario course (metadata rounded to 0.1 deg)' % scale_name, max(head_err) <= 0.05 + 1e-9,
              '%.3f deg' % max(head_err))
        check('W3 %s: track on the placed route' % scale_name, max(perp) < 1.0, 'max %.2f m' % max(perp))
        check('W3 %s: fixed track identical under hold and intercept' % scale_name, max(same) < 1e-9, '%.1e m' % max(same))
        check('W3 %s: every episode data-shaped' % scale_name, all(shaped), '%d of %d' % (sum(shaped), len(shaped)))
    table(['scale', 'episodes (hold)', 'largest yaw rate of the played heading [deg/s] P50 / P95 / max',
           'same from 1 s position chords (discretisation, information only)', 'limit', 'first heading error [deg]',
           'track to route [m] P50 / P95 / max', 'hold vs intercept track difference [m]', 'outcomes (hold)'], rows)


def w4_scenario_env(shapes, seeds=(5, 6, 7)):
    say('## W4 ScenarioAttackEnv (2 km arena) with the trace set')
    say('')
    say('Three families, %d seeds each, own ship holding course, 100 decisions per episode; yaw rate from the reported heading.' % len(seeds))
    say('')
    rows = []
    for trace_scale in (1.0, 35.0 / 244.74):
        env = ST.ScenarioAttackEnv(E.ownship(0, 0, 0, 0, 0, 0), scenario='mix', trace=TRACE, cycle_scenarios=True,
                                   trace_scale=trace_scale, seed=5, save_dir=os.path.join(OUT, 'env_tmp'))
        dt_sub = float(env.dt_sub)
        yaw, shaped, head_err = [], [], []
        for seed in seeds:
            env.seed(seed)
            env._cycle_i = 0
            for _ in range(len(env.families)):
                n, ev, track = run_episode(env, lambda e: 1, 100)
                yaw.append(played_yaw(env.target, dt_sub).max())
                shaped.append(ev['data_shaped'])
                head_err.append(abs(E.wrap_deg(env.current['target_course_deg'] - math.degrees(env.target.route_table['psi'][0]))))
        info = env.trace_info
        rows.append(['%.3f' % trace_scale, '%d / %d' % (info['n_accepted'], info['n_traces']), q3(yaw, '%.3f'), '%.3f' % env.turn_rate,
                     '%.3f' % max(head_err), '%d / %d' % (sum(shaped), len(shaped))])
        check('W4 trace_scale %.3f: yaw within the target turn rate' % trace_scale, max(yaw) <= env.turn_rate * 1.001,
              'largest %.3f vs %.3f deg/s' % (max(yaw), env.turn_rate))
        check('W4 trace_scale %.3f: first heading equals the scenario course (metadata rounded to 0.1 deg)' % trace_scale,
              max(head_err) <= 0.05 + 1e-9, '%.3f deg' % max(head_err))
    table(['trace_scale', 'accepted / traces', 'largest yaw rate of the played heading [deg/s] P50 / P95 / max', 'limit',
           'first heading error [deg]', 'data-shaped episodes'], rows)


def w5_old_vs_new(wins):
    say('## W5 The chain before step 4 against the wired chain, same windows, 6 m/s, unscaled')
    say('')
    old_yaw, old_jumps, new_yaw, new_jumps = [], [], 0, 0
    new_y = []
    for w in wins:
        old = ST.TargetShip(ST.rotate_translate(ST.smooth_path(w), (0.0, 0.0), 0.0), 6.0, automation='fixed')
        psi = [old.psi]
        for _ in range(int(ST.route_length(old.route) / 6.0) - 2):
            old.advance(1.0)
            psi.append(old.psi)
        y = np.abs(np.array([E.wrap_deg(v) for v in np.diff(psi)]))
        old_yaw.append(y.max()); old_jumps.append(int((y > 1.0).sum()))
        new = ST.TargetShip(A.place_route(A.build_route(w), (0.0, 0.0), 0.0), 6.0, automation='fixed')
        psi = [new.psi]
        for _ in range(int(new.route_table['s'][-1] / 6.0) - 2):
            new.advance(1.0)
            psi.append(new.psi)
        y = np.abs(np.array([E.wrap_deg(v) for v in np.diff(psi)]))
        new_y.append(y.max()); new_jumps += int((y > 1.0).sum())
    table(['chain', 'largest yaw rate of the playback [deg/s] P50 / P95 / max', 'heading steps above 1 deg/s (all routes)'],
          [['before step 4: box-5 + rotate_translate + segment playback', q3(old_yaw), int(np.sum(old_jumps))],
           ['wired: build_route + place_route + table playback', q3(new_y), new_jumps]])
    check('W5 wired chain has no heading steps above 1 deg/s', new_jumps == 0, '%d steps' % new_jumps)


def w6_no_trace():
    say('## W6 Without a trace: routes unchanged (straight chord) and the geometry self-test')
    say('')
    import check_attack_scenarios as C
    for scale_name in ('arena', 'maritime'):
        std = S.CollisionStandard(scale_name, 6.0)
        rng = np.random.RandomState(1)
        worst = 0.0
        for name in AS.SCENARIO_NAMES:
            sc = AS.make_attack_scenario(name, std, 6.0, rng, band='passing')
            r = np.asarray(sc['route'])
            assert sc['route_table'] is None and sc['meta']['data_shaped'] == 0 and len(r) == 2
            d = (r[1] - r[0]) / np.linalg.norm(r[1] - r[0])
            worst = max(worst, abs(E.wrap_deg(math.degrees(math.atan2(d[1], d[0])) - sc['meta']['course_diff_deg'])))
        check('W6 %s: straight chord along the scenario course (metadata rounded to 0.1 deg)' % scale_name, worst <= 0.05 + 1e-9,
              '%.3f deg' % worst)
        C.selftest(scale_name)
        check('W6 %s: check_attack_scenarios.selftest' % scale_name, True, 'all assertions hold')
    env = AS.EncounterAttackEnv(E.ownship(0, 0, 0, 0, 0, 0), scenario='mix', scale='arena', seed=3,
                                save_dir=os.path.join(OUT, 'env_tmp'))
    n, ev, track = run_episode(env, AS.intercept_action, 200)
    check('W6 arena episode without trace runs', ev['outcome'] in ('success', 'clear', 'off_map', 'unresolved'),
          '%d decisions, outcome %s, data_shaped %s' % (n, ev['outcome'], ev['data_shaped']))


def main():
    stamp = lambda p: datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime('%Y-%m-%d %H:%M:%S')
    say('# Check of step 4 of the mending plan for part 1: the route chain wired into the environments')
    say('')
    say('Run %s. Code as of: scenario_targets.py %s, attack_scenarios.py %s, ais_prep.py %s. Trace set: %s (%s).' % (
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), stamp(os.path.join(HERE, 'scenario_targets.py')),
        stamp(os.path.join(HERE, 'attack_scenarios.py')), stamp(os.path.join(HERE, 'ais_prep.py')),
        os.path.relpath(TRACE, HERE), stamp(TRACE)))
    say('')
    shapes = ST.load_trace_shapes(TRACE)
    wins = np.load(WXY)
    w1_loader(shapes, wins)
    prepared = w2_routes(shapes)
    w3_encounter_env(prepared)
    w4_scenario_env(shapes)
    w5_old_vs_new(wins)
    w6_no_trace()
    say('## Result')
    say('')
    say('%d checks failed.' % len(FAILS) if FAILS else 'All checks pass.')
    for f in FAILS:
        say('- ' + f)
    with open(os.path.join(OUT, 'results_env_wiring.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(LINES) + '\n')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
