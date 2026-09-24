"""
check_attack_scenarios.py -- checks of the attack-scenario set (attack_scenarios.py).

  1. geometry self-test: for every scenario and DCPA band the initial DCPA is the
     drawn value, the target passes on the named side, bow / stern crossings
     cross where they say, parallel lanes start abeam and astern
  2. scripted baselines per scenario x band: hold course (the attacker does
     nothing), intercept (collision-course heading), pursuit (steer at the
     target); outcome mix, event grade, time to the event
  3. catalogue figure of the initial geometries (chart_viz.catalogue), and
     optionally a grid GIF of one intercept episode per scenario

  python check_attack_scenarios.py --out runs/enc_check --bands passing close v2 --episodes 12 --grid
"""
import argparse
import csv
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import env_moving_obj as E            # noqa: E402
import encounter_standard as S        # noqa: E402
import attack_scenarios as A          # noqa: E402


def selftest(scale, n=40, seed=0):
    std = S.CollisionStandard(scale, 6.0)
    rng = np.random.RandomState(seed)
    checked = 0
    for name in A.SCENARIO_NAMES:
        spec = A.ATTACK_SCENARIOS[name]
        for band in A.BANDS:
            for _ in range(n):
                sc = A.make_attack_scenario(name, std, std.speed, rng, band=band)
                m, route = sc['meta'], np.asarray(sc['route'])
                d_t = (route[1] - route[0]) / np.linalg.norm(route[1] - route[0])
                p_t, v_t, v_o = route[0], d_t * std.speed, np.array([std.speed, 0.0])
                lo_hi = A.band_range(band, std)
                if spec['family'] == 'parallel':
                    assert np.sign(p_t[1]) == spec['side'] and p_t[0] < 0, (name, p_t)
                    if lo_hi:
                        assert lo_hi[0] - 1e-6 <= abs(p_t[1]) <= lo_hi[1] + 1e-6, (name, band, p_t)
                    assert abs(math.degrees(math.atan2(d_t[1], d_t[0]))) < 1e-6
                    checked += 1
                    continue
                assert abs(m['initial_dcpa_m'] - m['target_dcpa_m']) < 0.5, (name, band, m)
                if lo_hi:
                    assert lo_hi[0] - 0.5 <= m['initial_dcpa_m'] <= lo_hi[1] + 0.5, (name, band, m)
                tc = m['initial_tcpa_s']
                assert tc > 0, (name, band, m)
                rel = (p_t + v_t * tc) - v_o * tc                         # target minus attacker at the CPA
                if m['initial_dcpa_m'] > 1.0:
                    assert np.sign(rel[1]) == spec['side'], (name, band, rel)
                if spec['family'].startswith('crossing') and m['initial_dcpa_m'] > 1.0:
                    t0 = -p_t[1] / v_t[1]                                  # target crosses the attacker's track line
                    ahead = (p_t[0] + v_t[0] * t0) > std.speed * t0
                    assert ahead == name.endswith('_bow'), (name, band, t0, ahead)
                assert abs(abs(m['course_diff_deg']) - {'head_on': 180, 'crossing_starboard': 90,
                                                        'crossing_port': 90}[spec['family']]) <= 30.5
                checked += 1
    print('selftest %-8s %s: %d scenario draws OK' % (scale, std, checked))
    return std


def run_baselines(scale, band, episodes, policies, seed, out, control='full'):
    rows = []
    for name in A.SCENARIO_NAMES:
        env = A.EncounterAttackEnv(E.ownship(0, 0, 0, 0, 0, 0), scenario=name, dcpa_band=band, scale=scale,
                                   save_dir=out, seed=seed, control=control)
        for pol in policies:
            env.seed(seed)
            for ep in range(episodes):
                obs = env.reset()
                done = False
                while not done:
                    obs, r, done, flag = env.step(A.BASELINES[pol](env, obs))
                ev = env.evaluation()
                ev['policy'] = pol
                rows.append(ev)
    return rows


def summarise(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r['scenario'], r['band'], r['policy']), []).append(r)
    table = []
    for (scen, band, pol), g in groups.items():
        oc = [r['outcome'] for r in g]
        succ = [r for r in g if r['success']]
        table.append(dict(scenario=scen, band=band, policy=pol, n=len(g),
                          success=round(np.mean([r['success'] for r in g]), 2),
                          clear=round(oc.count('clear') / len(g), 2), off_map=round(oc.count('off_map') / len(g), 2),
                          unresolved=round(oc.count('unresolved') / len(g), 2),
                          collision=sum(1 for r in g if r['termination'] == 'collision'),
                          mean_steps=round(np.mean([r['steps'] for r in g]), 1),
                          t_event_s=round(np.mean([r['steps'] * r['dt'] for r in succ]), 0) if succ else '',
                          min_range_m=round(np.median([r['min_distance'] for r in g]), 0),
                          grade_hist='/'.join(str(sum(1 for r in g if r['event_grade'] == k)) for k in range(4)),
                          event_score=round(np.mean([r['event_score'] for r in g]), 1),
                          initial_dcpa_m=round(np.mean([r['initial_dcpa_m'] for r in g]), 0),
                          initial_range_m=round(np.mean([r['initial_range_m'] for r in g]), 0),
                          max_speed=round(np.mean([r['max_speed'] for r in g]), 2)))
    return table


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default='runs/enc_check')
    p.add_argument('--scale', default='arena')
    p.add_argument('--control', default='full', choices=['full', 'turn'])
    p.add_argument('--bands', nargs='+', default=['passing'])
    p.add_argument('--episodes', type=int, default=12)
    p.add_argument('--policies', nargs='+', default=['hold', 'intercept', 'pursuit'])
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--grid', action='store_true', help='grid GIF of one intercept episode per scenario (first band)')
    p.add_argument('--no_selftest', action='store_true')
    a = p.parse_args()
    out = a.out if os.path.isabs(a.out) else os.path.join(HERE, a.out)
    os.makedirs(out, exist_ok=True)
    if not a.no_selftest:
        for sc in ('arena', 'maritime'):
            selftest(sc)
    import chart_viz
    std = S.CollisionStandard(a.scale, 6.0)
    all_rows, table = [], []
    for band in a.bands:
        t0 = time.time()
        rows = run_baselines(a.scale, band, a.episodes, a.policies, a.seed, out, a.control)
        all_rows += rows
        tb = summarise(rows)
        table += tb
        print('\nband %s (%d episodes, %.1f s)' % (band, len(rows), time.time() - t0))
        print('%-25s %-9s %5s %5s %5s %5s %5s %6s %6s %9s %7s %5s' % ('scenario', 'policy', 'succ', 'clear', 'offm', 'unres',
                                                                  'coll', 't_ev', 'minR', 'grades', 'score', 'vmax'))
        for r in tb:
            print('%-25s %-9s %5.2f %5.2f %5.2f %5.2f %5d %6s %6.0f %9s %7.1f' % (
                r['scenario'], r['policy'], r['success'], r['clear'], r['off_map'], r['unresolved'], r['collision'],
                r['t_event_s'], r['min_range_m'], r['grade_hist'], r['event_score']) + ' %5.1f' % r['max_speed'])
        rng = np.random.RandomState(a.seed)
        samples = {name: [A.make_attack_scenario(name, std, std.speed, rng, band=band) for _ in range(6)]
                   for name in A.SCENARIO_NAMES}
        path = chart_viz.catalogue(samples, os.path.join(out, 'catalogue_%s_%s.png' % (a.scale, band)), std.as_dict(), band)
        print('wrote', path)
    write_csv(os.path.join(out, 'baseline_episodes.csv'), all_rows)
    write_csv(os.path.join(out, 'baseline_summary.csv'), table)
    with open(os.path.join(out, 'baseline_summary.md'), 'w', encoding='utf-8') as f:
        cols = ['scenario', 'band', 'policy', 'n', 'success', 'clear', 'off_map', 'unresolved', 'collision', 't_event_s',
                'min_range_m', 'grade_hist', 'event_score', 'initial_dcpa_m', 'initial_range_m', 'max_speed']
        f.write('| ' + ' | '.join(cols) + ' |\n|' + '---|' * len(cols) + '\n')
        for r in table:
            f.write('| ' + ' | '.join(str(r[c]) for c in cols) + ' |\n')
    print('wrote', os.path.join(out, 'baseline_summary.csv'))
    if a.grid:
        recs = []
        for name in A.SCENARIO_NAMES:
            env = A.EncounterAttackEnv(E.ownship(0, 0, 0, 0, 0, 0), scenario=name, dcpa_band=a.bands[0], scale=a.scale,
                                       save_dir=out, seed=a.seed, control=a.control)
            recs.append(chart_viz.record_episode(env, policy=A.intercept_action))
        gp = chart_viz.animate_grid(recs, os.path.join(out, 'intercept_grid_%s.gif' % a.bands[0]), fps=10)
        sb = chart_viz.storyboard(recs[2], os.path.join(out, 'intercept_%s_storyboard.png' % recs[2]['scenario']))
        ap = chart_viz.animate_episode(recs[2], os.path.join(out, 'intercept_%s.gif' % recs[2]['scenario']), fps=10,
                                       label='scripted intercept')
        print('wrote', gp, sb, ap)


if __name__ == '__main__':
    main()
