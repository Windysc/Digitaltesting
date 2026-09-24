"""
viz_tool.py -- visualisation for PPO scenario-generation runs
(main_attack_ppo_enc.py / main_attack_ppo_scen.py with the rebuilt MassTestingEnv).

Sub-commands, all writing under <run>/viz/ (<run> = the --save_dir of a
training run: args.json, train_logs/, eval_logs/, models/):

  curves  <run> [<run2> ...]   training return / episode length / success and the
                               evaluation curves; extra runs are overlaid on the
                               evaluation panels (e.g. CPU vs GPU, seeds)
  evals   <run>                per-evaluation outcome mix, final positions on the map,
                               closest-approach histograms (from eval_dicts_*.csv)
  replay  <run> [--ckpt best|last|episodeN|<path>] [--episodes N] [--deterministic]
                               rebuild the scenario from args.json, roll the checkpoint
                               out (sampling actions like the scripts' own eval(); use
                               --deterministic for argmax), save a trajectory fan PNG, an
                               animated GIF of the first episode and a per-episode CSV
  chart   <run> [--episodes N] [--grid M]
                               chart-style animations (navigation-display look of the v2
                               build: full-scope map, detail view with hulls, vectors, COLREG
                               sectors, CPA prediction, data box, side panels), one GIF and
                               storyboard per episode under <run>/viz/chart/, optional grid GIF
  report  <run>                one HTML page embedding everything above
  all     <run> [replay opts]  curves + evals + replay + report
  compare <run> <run2> ...     overlay the evaluation curves of several runs, write
                               compare.png / compare.csv / compare.html (default into
                               <parent of first run>/compare, or --out DIR)

Examples
  python viz_tool.py all  runs/attack_gpu --episodes 10
  python viz_tool.py curves runs/nav_cpu runs/nav_gpu
"""
import argparse
import base64
import csv
import glob
import io
import json
import math
import os
import re
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import env_moving_obj as E                     # noqa: E402
import env_moving_attack as A                  # noqa: E402
from PPO import PPO                            # noqa: E402
import chart_viz                               # noqa: E402  chart-style renderer (v2 look)

OUTCOME_COLOURS = {'success': 'tab:green', 'collision': 'tab:red', 'off_map': 'tab:orange',
                   'timeout': 'tab:grey', 'running': 'tab:blue', 'clear': 'tab:purple', 'unresolved': 'tab:grey'}


# ------------------------------------------------------------------ scenarios
def load_args(run):
    with open(os.path.join(run, 'args.json'), encoding='utf-8') as f:
        return json.load(f)


def infer_task(args, override=None):
    if override:
        return override
    return 'attack' if 'attack' in str(args.get('reward_type', '')) else 'navigate'


def _rng(text):
    if not text:
        return None
    return tuple(float(v) for v in str(text).split(','))


def build_env(run, task=None, save_dir=None):
    args = load_args(run)
    task = infer_task(args, task)
    if 'dcpa_band' in args:                     # main_attack_ppo_enc.py run (collision standard, lifecycle)
        from attack_scenarios import EncounterAttackEnv
        env = EncounterAttackEnv(E.ownship(0, 0, 0, 0, 0, 0), scenario=args['scenario'], dcpa_band=args['dcpa_band'],
                                 scale=args.get('scale', 'arena'), cruise_speed=args.get('cruise_speed', 6.0),
                                 turn_rate_deg=(args.get('turn_rate') or None), trace=args.get('trace') or None,
                                 success_severity=args.get('success_severity', 2), hold_steps=args.get('hold_steps', 2),
                                 max_encounter_s=(args.get('max_encounter_s') or None), cycle_scenarios=True,
                                 control=args.get('control', 'turn'), v_range=_rng(args.get('speed_range')),
                                 a_range=_rng(args.get('accel_range')), cog_limit=args.get('cog_limit', 180.0),
                                 long_range=_rng(args.get('long_range')), lat_range=_rng(args.get('lat_range')),
                                 acc_levels=args.get('acc_levels', 3), rot_levels=args.get('rot_levels', 3),
                                 decision_interval=args.get('decision_interval', 600),
                                 reward_type=args.get('reward_type', 'final_attack_reward'),
                                 save_dir=save_dir or os.path.join(run, 'viz'), seed=args.get('seed', 0) + 1000)
        return env, args, 'attack'
    if 'scenario' in args:                      # main_attack_ppo_scen.py run
        from scenario_targets import ScenarioAttackEnv
        env = ScenarioAttackEnv(E.ownship(0, 0, 0, 0, 0, 0), scenario=args['scenario'],
                                cruise_speed=args.get('cruise_speed', 6.0), trace=args.get('trace') or None,
                                trace_scale=args.get('trace_scale', 1.0),
                                automation=args.get('automation', 'manual'), cycle_scenarios=True,
                                turn_rate_deg=args.get('turn_rate', 1.0), speed_control=bool(args.get('speed_control', 0)),
                                duration=args.get('duration', 60000), decision_interval=args.get('decision_interval', 600),
                                reward_type=args.get('reward_type', 'final_attack_reward'),
                                X_LEN=args.get('map_x_size', 2000), Y_LEN=args.get('map_y_size', 1000),
                                save_dir=save_dir or os.path.join(run, 'viz'), seed=args.get('seed', 0) + 1000,
                                attack_range=(args.get('attack_range') or None))
        return env, args, 'attack'
    raise SystemExit('%s: args.json has neither "dcpa_band" (main_attack_ppo_enc.py) nor "scenario" '
                     '(main_attack_ppo_scen.py); runs of the obstacle-scene trainers removed on 2026-09-24 '
                     'cannot be rebuilt' % run)


def find_checkpoint(run, args, which='best'):
    if os.path.isfile(which):
        return which
    d = os.path.join(run, 'models', 'MassTestingEnv')
    seed = args.get('seed', 0)
    base = os.path.join(d, 'PPO_MassTestingEnv_seed%d' % seed)
    cands = {'last': base + '.pth', 'best': base + '_best.pth'}
    m = re.match(r'episode(\d+)$', which)
    if m:
        cands[which] = base + '_episode%s.pth' % m.group(1)
    path = cands.get(which)
    if path and os.path.isfile(path):
        return path
    if which == 'best' and os.path.isfile(cands['last']):
        print('no best checkpoint yet, using last:', cands['last'])
        return cands['last']
    avail = sorted(glob.glob(os.path.join(d, '*.pth')))
    raise SystemExit('checkpoint %r not found; available: %s' % (which, avail))


# ------------------------------------------------------------------- logs
def read_train_log(run):
    path = os.path.join(run, 'train_logs', 'train_result.txt')
    ep, ret, steps, succ = [], [], [], []
    with open(path, encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.split()
            if len(p) < 4:
                continue
            ep.append(int(p[0])); ret.append(float(p[1])); steps.append(float(p[2])); succ.append(p[3] == 'True')
    return np.asarray(ep), np.asarray(ret), np.asarray(steps), np.asarray(succ, dtype=float)


def read_eval_log(run):
    path = os.path.join(run, 'eval_logs', 'eval_result.txt')
    rows = []
    with open(path, encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.split()
            if len(p) >= 4:
                rows.append([float(v) for v in p[:4]])
    arr = np.asarray(rows) if rows else np.zeros((0, 4))
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


def read_eval_dicts(run):
    out = {}
    for path in glob.glob(os.path.join(run, 'eval_logs', 'eval_dicts_episode_*.csv')):
        ep = int(re.search(r'episode_(\d+)', path).group(1))
        with open(path, encoding='utf-8') as f:
            out[ep] = list(csv.DictReader(f))
    return dict(sorted(out.items()))


def moving_average(x, w):
    if len(x) < w:
        return np.asarray(x, dtype=float)
    c = np.cumsum(np.insert(np.asarray(x, dtype=float), 0, 0.0))
    ma = (c[w:] - c[:-w]) / w
    return np.concatenate([np.full(w - 1, np.nan), ma])


def viz_dir(run):
    d = os.path.join(run, 'viz')
    os.makedirs(d, exist_ok=True)
    return d


# ----------------------------------------------------------------- curves
def cmd_curves(runs):
    run = runs[0]
    out = viz_dir(run)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ep, ret, steps, succ = read_train_log(run)
    w = max(5, len(ep) // 50)
    ax = axes[0, 0]
    ax.plot(ep, ret, '.', ms=2, alpha=0.3, color='tab:blue')
    ax.plot(ep, moving_average(ret, w), color='tab:blue', label='moving avg (%d)' % w)
    ax.set_title('training episode return'); ax.set_xlabel('episode'); ax.legend()
    ax = axes[0, 1]
    ax.plot(ep, steps, '.', ms=2, alpha=0.3, color='tab:purple')
    ax.plot(ep, moving_average(steps, w), color='tab:purple')
    ax.set_title('training episode length (decisions)'); ax.set_xlabel('episode')
    ax = axes[0, 2]
    ax.plot(ep, moving_average(succ, w), color='tab:green')
    ax.set_ylim(-0.02, 1.02); ax.set_title('training success rate (moving avg %d)' % w); ax.set_xlabel('episode')
    for r in runs:
        label = os.path.basename(os.path.normpath(r))
        e_ep, e_ret, e_len, e_succ = read_eval_log(r)
        if len(e_ep) == 0:
            continue
        axes[1, 0].plot(e_ep, e_succ, 'o-', label=label)
        axes[1, 1].plot(e_ep, e_ret, 'o-', label=label)
        axes[1, 2].plot(e_ep, e_len, 'o-', label=label)
    axes[1, 0].set_ylim(-0.02, 1.02); axes[1, 0].set_title('evaluation success rate'); axes[1, 0].set_xlabel('episode')
    axes[1, 1].set_title('evaluation mean return'); axes[1, 1].set_xlabel('episode')
    axes[1, 2].set_title('evaluation mean length (decisions)'); axes[1, 2].set_xlabel('episode')
    for ax in axes[1]:
        ax.legend(fontsize=8)
    args = load_args(run)
    scene = ('scenario ' + str(args.get('scenario'))) if 'scenario' in args else 'obst %s dest %s' % (args.get('obst_id'), args.get('dest_id'))
    fig.suptitle('%s | %s | %s | seed %s' % (label_of(run), args.get('reward_type'), scene, args.get('seed')))
    fig.tight_layout()
    path = os.path.join(out, 'curves.png')
    fig.savefig(path, dpi=110); plt.close(fig)
    print('wrote', path)
    return path


def label_of(run):
    return os.path.basename(os.path.normpath(run))


# ------------------------------------------------------------------ evals
def cmd_evals(run):
    out = viz_dir(run)
    dicts = read_eval_dicts(run)
    if not dicts:
        print('no eval_dicts_*.csv in', run)
        return None
    env, args, task = build_env(run)
    eps = list(dicts)
    outcomes = sorted({r['outcome'] for rows in dicts.values() for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    # outcome mix per evaluation point
    ax = axes[0]
    bottom = np.zeros(len(eps))
    for oc in outcomes:
        frac = np.array([np.mean([r['outcome'] == oc for r in dicts[e]]) for e in eps])
        ax.bar([str(e) for e in eps], frac, bottom=bottom, color=OUTCOME_COLOURS.get(oc, 'k'), label=oc)
        bottom += frac
    ax.set_title('outcome mix per evaluation'); ax.set_xlabel('training episode'); ax.legend(fontsize=8)
    # final positions of the LAST evaluation on the map
    ax = axes[1]
    draw_static(ax, env, task)
    last = dicts[eps[-1]]
    for r in last:
        if 'final_x' not in r:                      # records written before the field existed
            continue
        ax.plot(float(r['final_x']), float(r['final_y']), 'o', ms=5,
                color=OUTCOME_COLOURS.get(r['outcome'], 'k'), alpha=0.8)
    for oc in outcomes:
        ax.plot([], [], 'o', color=OUTCOME_COLOURS.get(oc, 'k'), label=oc)
    ax.legend(fontsize=8, loc='upper left'); ax.set_title('final positions, evaluation at episode %d' % eps[-1])
    # closest approach histogram per evaluation point
    ax = axes[2]
    key = 'min_target_dist' if task == 'attack' else 'dist_to_destination'
    for e in eps:
        vals = [float(r[key]) for r in dicts[e] if float(r[key]) >= 0]
        if vals:
            ax.hist(vals, bins=20, alpha=0.5, label='episode %d' % e)
    ax.set_xlabel(key + ' [m]'); ax.set_title('closest approach' if task == 'attack' else 'final distance to destination')
    ax.legend(fontsize=8)
    fig.suptitle('%s | evaluation records (%d per point)' % (label_of(run), len(last)))
    fig.tight_layout()
    path = os.path.join(out, 'evals.png')
    fig.savefig(path, dpi=110); plt.close(fig)
    print('wrote', path)
    return path


# ----------------------------------------------------------------- replay
def draw_static(ax, env, task, t_end=None):
    ax.add_patch(plt.Rectangle((0, -env.Y_LEN), env.X_LEN, 2 * env.Y_LEN, fill=False, ls='--', color='grey'))
    per_episode = getattr(env, 'per_episode_targets', False)
    for o in ([] if per_episode else env.objects):
        is_t = o in env.targets
        colour = 'tab:red' if is_t else 'tab:brown'
        ax.add_patch(Polygon(o.corners(0.0), closed=True, fc=colour, alpha=0.3, ec=colour))
        ax.add_patch(Circle(o.position(0.0), o.risk_range, fill=False, ls=':', color=colour, alpha=0.6))
        if o.is_moving and t_end:
            x0, y0 = o.position(0.0); x1, y1 = o.position(t_end)
            ax.plot([x0, x1], [y0, y1], ':', color=colour, alpha=0.6)
        ax.annotate('%s %s' % ('target' if is_t else 'obst', o.id), o.position(0.0), fontsize=8, color=colour)
    if task == 'navigate':
        nt = env.nt
        ax.add_patch(Circle((nt.x, nt.y), nt.target_deviation_distance, fill=False, color='tab:green'))
        rad = math.radians(nt.direction)
        ax.arrow(nt.x, nt.y, 150 * math.cos(rad), 150 * math.sin(rad), head_width=40, color='tab:green')
    pad = max(env.X_LEN, 2 * env.Y_LEN) * 0.6 if per_episode else E.MAP_MARGIN + 200
    ax.set_xlim(-pad, env.X_LEN + pad)
    ax.set_ylim(-env.Y_LEN - pad, env.Y_LEN + pad)
    ax.set_aspect('equal'); ax.set_xlabel('long [m]'); ax.set_ylabel('lat [m]')


def fan_plot(records, env, task, path, title):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    t_end = max(r['dt'] * (len(r['own']) - 1) for r in records)
    draw_static(ax, env, task, t_end=t_end)
    seen = set()
    per_episode = getattr(env, 'per_episode_targets', False)
    for r in records:
        if per_episode and r.get('target_track') is not None:
            tt = r['target_track']
            ax.plot(tt[:, 0], tt[:, 1], '-', color='tab:red', lw=0.8, alpha=0.5)
            ax.plot(tt[0, 0], tt[0, 1], 'o', color='tab:red', ms=3, alpha=0.6)
        oc = r['eval']['outcome']
        ax.plot(r['path'][:, 0], r['path'][:, 1], '-', lw=1.2, alpha=0.8, color=OUTCOME_COLOURS.get(oc, 'k'),
                label=oc if oc not in seen else None)
        seen.add(oc)
        ax.plot(r['path'][-1, 0], r['path'][-1, 1], 's', ms=4, color=OUTCOME_COLOURS.get(oc, 'k'))
    ax.plot(records[0]['path'][0, 0], records[0]['path'][0, 1], 'k^', ms=8, label='own start')
    if per_episode:
        ax.plot([], [], '-', color='tab:red', lw=0.8, label='target tracks (start marked)')
    ax.legend(fontsize=8, loc='upper left'); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    print('wrote', path)


def cmd_replay(run, ckpt='best', episodes=10, stochastic=True, task=None, fps=8):
    out = viz_dir(run)
    env, args, task = build_env(run, task)
    ckpt_path = find_checkpoint(run, args, ckpt)
    agent = PPO(env.observation_space.shape[0], env.action_space.n, 3e-4, 1e-3, 0.99, 80, 0.2, False, 0.6)
    agent.load(ckpt_path)
    env.seed(args.get('seed', 0))
    records = []
    t0 = time.time()
    for i in range(episodes):
        records.append(chart_viz.record_episode(env, agent, deterministic=not stochastic))
    dt = time.time() - t0
    mode = 'stochastic' if stochastic else 'deterministic'
    n_succ = sum(r['eval']['success'] for r in records)
    if getattr(env, 'per_episode_targets', False):
        per = {}
        for r in records:
            per.setdefault(r.get('scenario', '?'), []).append(r['eval']['success'])
        print('replay per scenario:', {k: '%d/%d' % (sum(v), len(v)) for k, v in sorted(per.items())})
    title = '%s | %s | %s policy | %d/%d success' % (label_of(run), os.path.basename(ckpt_path), mode, n_succ, episodes)
    fan_plot(records, env, task, os.path.join(out, 'replay_fan.png'), title)
    chart_viz.animate_episode(records[0], os.path.join(out, 'replay_episode1.gif'), fps=fps, label=label_of(run))
    chart_viz.storyboard(records[0], os.path.join(out, 'replay_episode1_storyboard.png'))
    csv_path = os.path.join(out, 'replay_episodes.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]['eval'].keys()))
        w.writeheader()
        for r in records:
            w.writerow(r['eval'])
    print('wrote', csv_path)
    print('replay: %d episodes in %.1f s, %s policy, checkpoint %s, success %d/%d'
          % (episodes, dt, mode, ckpt_path, n_succ, episodes))
    return records


# ------------------------------------------------------------------ chart
def cmd_chart(run, ckpt='best', episodes=3, grid=0, stochastic=True, task=None, fps=8):
    """Chart-style animations (v2 look) for several episodes, plus storyboards and a grid GIF."""
    out = os.path.join(viz_dir(run), 'chart')
    os.makedirs(out, exist_ok=True)
    env, args, task = build_env(run, task)
    ckpt_path = find_checkpoint(run, args, ckpt)
    agent = PPO(env.observation_space.shape[0], env.action_space.n, 3e-4, 1e-3, 0.99, 80, 0.2, False, 0.6)
    agent.load(ckpt_path)
    env.seed(args.get('seed', 0) + 2000)
    summary = []
    for i in range(episodes):
        rec = chart_viz.record_episode(env, agent, deterministic=not stochastic)
        base = os.path.join(out, 'episode%d_%s' % (i + 1, rec['scenario'] or task))
        chart_viz.animate_episode(rec, base + '.gif', fps=fps, label=label_of(run))
        chart_viz.storyboard(rec, base + '_storyboard.png')
        ev = dict(rec['eval']); ev['files'] = [base + '.gif', base + '_storyboard.png']
        summary.append(ev)
        print('episode %d [%s / %s]: %s in %d decisions, closest %.0f m -> %s.gif' % (
            i + 1, rec['scenario'], rec['automation'], rec['outcome'], rec['steps'],
            rec['dh'].min() if rec['has_target'] else 0.0, base))
    if grid > 0:
        recs = [chart_viz.record_episode(env, agent, deterministic=not stochastic) for _ in range(grid)]
        gp = os.path.join(out, 'grid%d.gif' % grid)
        chart_viz.animate_grid(recs, gp, fps=fps)
        summary.append(dict(grid=[r['eval'] for r in recs], files=[gp]))
        print('grid: %s  (%s)' % (gp, ', '.join('%s:%s' % (r['scenario'][:10], r['outcome']) for r in recs)))
    with open(os.path.join(out, 'chart_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=float)
    return summary


# ----------------------------------------------------------------- report
def _b64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('ascii')


def _img(path, alt):
    if not path or not os.path.isfile(path):
        return '<p><em>%s not available</em></p>' % alt
    ext = 'gif' if path.endswith('.gif') else 'png'
    return '<img alt="%s" style="max-width:100%%" src="data:image/%s;base64,%s">' % (alt, ext, _b64(path))


def _table(rows, cols):
    h = '<table><tr>' + ''.join('<th>%s</th>' % c for c in cols) + '</tr>'
    for r in rows:
        h += '<tr>' + ''.join('<td>%s</td>' % r.get(c, '') for c in cols) + '</tr>'
    return h + '</table>'


def cmd_report(run):
    out = viz_dir(run)
    args = load_args(run)
    task = infer_task(args)
    e_ep, e_ret, e_len, e_succ = read_eval_log(run)
    ep, ret, steps, succ = read_train_log(run)
    eval_rows = [dict(episode=int(a), success=round(b, 3), mean_return=round(c, 3), mean_length=round(d, 1))
                 for a, c, d, b in zip(e_ep, e_ret, e_len, e_succ)]
    replay_rows = []
    rp = os.path.join(out, 'replay_episodes.csv')
    if os.path.isfile(rp):
        with open(rp, encoding='utf-8') as f:
            replay_rows = list(csv.DictReader(f))
    html = io.StringIO()
    html.write('<!doctype html><html><head><meta charset="utf-8"><title>%s report</title>' % label_of(run))
    html.write('<style>body{font-family:system-ui,sans-serif;margin:24px;max-width:1400px}'
               'table{border-collapse:collapse;font-size:13px}th,td{border:1px solid #ccc;padding:3px 8px}'
               'h2{margin-top:32px}code{background:#f3f3f3;padding:1px 4px}</style></head><body>')
    if 'dcpa_band' in args:
        scene = ('attack scenarios %s, initial DCPA band %s, scale %s (v2 collision standard, v3 lifecycle), '
                 'fixed-track target' % (args.get('scenario'), args.get('dcpa_band'), args.get('scale')))
    elif 'scenario' in args:
        scene = 'scenario %s, cruise %s m/s, target %s' % (args.get('scenario'), args.get('cruise_speed'), args.get('automation'))
    else:
        scene = 'obstacle %s, destination %s' % (args.get('obst_id'), args.get('dest_id'))
    html.write('<h1>%s</h1><p>task <b>%s</b> | reward <code>%s</code> | %s '
               '| seed %s | %d training episodes | generated %s</p>'
               % (label_of(run), task, args.get('reward_type'), scene,
                  args.get('seed'), len(ep), time.strftime('%Y-%m-%d %H:%M')))
    html.write('<h2>Training and evaluation curves</h2>' + _img(os.path.join(out, 'curves.png'), 'curves'))
    html.write('<h2>Evaluation records</h2>' + _img(os.path.join(out, 'evals.png'), 'evals'))
    html.write('<h3>Evaluation summary</h3>' + _table(eval_rows, ['episode', 'success', 'mean_return', 'mean_length']))
    html.write('<h2>Checkpoint replay</h2>' + _img(os.path.join(out, 'replay_fan.png'), 'trajectory fan'))
    html.write('<h3>Episode 1 storyboard</h3>' + _img(os.path.join(out, 'replay_episode1_storyboard.png'), 'storyboard'))
    html.write('<h3>Episode 1 animation (chart display)</h3>' + _img(os.path.join(out, 'replay_episode1.gif'), 'animation'))
    if replay_rows:
        extra = (['scenario', 'band', 'initial_dcpa_m', 'termination', 'max_severity', 'event_score'] if 'dcpa_band' in args
                 else ['scenario', 'dcpa_offset_m', 't_meet_s', 'target_evasions'] if 'scenario' in args else [])
        cols = ['episode'] + extra + ['outcome', 'steps', 'sim_time_s', 'min_target_dist' if task == 'attack' else 'dist_to_destination',
                'min_hazard_dist', 'steps_in_risk', 'path_length', 'mean_speed', 'total_reward']
        html.write('<h3>Replay episodes</h3>' + _table(replay_rows, cols))
    html.write('<h2>Run arguments</h2><pre>%s</pre>' % json.dumps(args, indent=2))
    html.write('</body></html>')
    path = os.path.join(out, 'report.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html.getvalue())
    print('wrote', path)
    return path


# ---------------------------------------------------------------- compare
def cmd_compare(runs, out=None):
    """Overlay the evaluation curves of several runs and tabulate them."""
    out = out or os.path.join(os.path.dirname(os.path.normpath(runs[0])), 'compare')
    os.makedirs(out, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
    rows = []
    for r in runs:
        if not os.path.isfile(os.path.join(r, 'args.json')):
            continue
        label, args = label_of(r), load_args(r)
        e_ep, e_ret, e_len, e_succ = read_eval_log(r)
        ep, ret, steps, succ = read_train_log(r)
        if len(e_ep):
            axes[0, 0].plot(e_ep, e_succ, 'o-', ms=4, label=label)
            axes[0, 1].plot(e_ep, e_ret, 'o-', ms=4, label=label)
            axes[1, 0].plot(e_ep, e_len, 'o-', ms=4, label=label)
        w = max(10, len(ep) // 30)
        axes[1, 1].plot(ep, moving_average(succ, w), label=label)
        k = int(np.argmax(e_succ)) if len(e_succ) else -1
        rows.append(dict(run=label, task=infer_task(args), obst_id=args.get('obst_id'), dest_id=args.get('dest_id'),
                         reward_type=args.get('reward_type'), episodes=int(len(ep)),
                         best_eval_success=round(float(e_succ[k]), 3) if k >= 0 else '',
                         best_at_episode=int(e_ep[k]) if k >= 0 else '',
                         final_eval_success=round(float(e_succ[-1]), 3) if len(e_succ) else '',
                         final_eval_return=round(float(e_ret[-1]), 3) if len(e_ret) else '',
                         final_eval_length=round(float(e_len[-1]), 1) if len(e_len) else '',
                         train_success_last100=round(float(np.mean(succ[-100:])), 3) if len(succ) else '',
                         train_return_last100=round(float(np.mean(ret[-100:])), 3) if len(ret) else ''))
    axes[0, 0].set_title('evaluation success rate'); axes[0, 0].set_ylim(-0.02, 1.02)
    axes[0, 1].set_title('evaluation mean return')
    axes[1, 0].set_title('evaluation mean length (decisions)')
    axes[1, 1].set_title('training success rate (moving average)'); axes[1, 1].set_ylim(-0.02, 1.02)
    for ax in axes.flat:
        ax.set_xlabel('training episode'); ax.legend(fontsize=8)
    fig.suptitle('comparison of %d runs' % len(rows))
    fig.tight_layout()
    png = os.path.join(out, 'compare.png')
    fig.savefig(png, dpi=110); plt.close(fig)
    cols = list(rows[0].keys()) if rows else []
    csv_path = os.path.join(out, 'compare.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    html = ('<!doctype html><html><head><meta charset="utf-8"><title>run comparison</title>'
            '<style>body{font-family:system-ui,sans-serif;margin:24px}table{border-collapse:collapse;font-size:13px}'
            'th,td{border:1px solid #ccc;padding:3px 8px}</style></head><body><h1>Run comparison</h1>'
            + _table(rows, cols) + _img(png, 'comparison') + '</body></html>')
    html_path = os.path.join(out, 'compare.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print('wrote', png); print('wrote', csv_path); print('wrote', html_path)
    return rows


# ------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['curves', 'evals', 'replay', 'report', 'all', 'compare', 'chart'])
    p.add_argument('runs', nargs='+', help='run directory (the --save_dir of the training script)')
    p.add_argument('--ckpt', default='best', help='best | last | episodeN | path (replay)')
    p.add_argument('--episodes', type=int, default=10, help='replay episodes')
    p.add_argument('--deterministic', action='store_true', help='argmax actions instead of sampling (replay)')
    p.add_argument('--task', choices=['navigate', 'attack'], default=None, help='override the task inferred from reward_type')
    p.add_argument('--fps', type=int, default=8)
    p.add_argument('--grid', type=int, default=0, help='chart: also animate this many episodes in one grid GIF')
    p.add_argument('--out', default=None, help='output directory for compare (default: <parent of first run>/compare)')
    a = p.parse_args(argv)
    run = a.runs[0]
    if a.command == 'curves':
        cmd_curves(a.runs)
    elif a.command == 'evals':
        cmd_evals(run)
    elif a.command == 'replay':
        cmd_replay(run, a.ckpt, a.episodes, not a.deterministic, a.task, a.fps)
    elif a.command == 'report':
        cmd_report(run)
    elif a.command == 'compare':
        cmd_compare(a.runs, a.out)
    elif a.command == 'chart':
        cmd_chart(run, a.ckpt, a.episodes, a.grid, not a.deterministic, a.task, a.fps)
    else:
        cmd_curves(a.runs); cmd_evals(run)
        cmd_replay(run, a.ckpt, a.episodes, not a.deterministic, a.task, a.fps)
        cmd_report(run)


if __name__ == '__main__':
    main()
