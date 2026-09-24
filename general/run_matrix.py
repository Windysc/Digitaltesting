"""
run_matrix.py -- batch runner for scenario x reward x seed matrices of the attack
trainers (main_attack_ppo_enc.py, task enc, the default; main_attack_ppo_scen.py,
task scen), with seed aggregation.

  python run_matrix.py --out runs/enc_full --tasks enc --control full --enc_scenarios mix --bands passing \
                       --seeds 1 2 3 --episodes 3000 --eval_every 250 --num_eval 24
  python run_matrix.py --out runs/enc_full --summary-only        # re-aggregate without training

Jobs are skipped when <out>/<name>/eval_logs/eval_result.txt already reaches the
requested episode count, so an interrupted matrix can be resumed.  Each job's
stdout goes to <out>/<name>/stdout.log.

Outputs in <out>/:  matrix_runs.csv (one row per run), matrix_summary.csv (mean and
sd over seeds per scenario x reward), matrix_success.png / matrix_length.png (eval
curves per scenario, seeds as thin lines, mean as thick line, colour = reward).
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

SCEN_SCENARIOS = ['head_on', 'crossing_starboard', 'crossing_port', 'mix']   # main_attack_ppo_scen.py
ENC_SCENARIOS = ['mix']                                                     # main_attack_ppo_enc.py
ATK_REWARDS = ['final_attack_reward', 'dense_attack_reward']
REWARD_COLOUR = {'sparse': 'tab:orange', 'dense': 'tab:blue'}


def family(reward_type):
    return 'sparse' if str(reward_type).startswith('final') else 'dense'


def make_jobs(args):
    jobs = []
    if 'enc' in args.tasks:
        for scen in (args.enc_scenarios or ENC_SCENARIOS):
            for band in args.bands:
                for reward in ATK_REWARDS:
                    for seed in args.seeds:
                        name = 'enc_%s_%s%s_%s_s%d' % (scen.replace(',', '+'), band,
                                                      '_full' if args.control == 'full' else '', family(reward), seed)
                        jobs.append(dict(name=name, script='main_attack_ppo_enc.py', enc_scenario=scen, band=band,
                                         reward=reward, seed=seed))
    if 'scen' in args.tasks:
        for scen in (args.scenarios or SCEN_SCENARIOS):
            for auto in args.automation:
                for reward in ATK_REWARDS:
                    for seed in args.seeds:
                        name = 'scen_%s_%s_%s_s%d' % (scen, auto, family(reward), seed)
                        jobs.append(dict(name=name, script='main_attack_ppo_scen.py', scenario=scen, automation=auto,
                                         reward=reward, seed=seed))
    return jobs


def is_done(run_dir, episodes):
    path = os.path.join(run_dir, 'eval_logs', 'eval_result.txt')
    if not os.path.isfile(path):
        return False
    with open(path, encoding='utf-8') as f:
        lines = [l.split() for l in f if l.strip()]
    return len(lines) > 1 and lines[-1][0].isdigit() and int(lines[-1][0]) >= episodes


def run_job(job, args):
    run_dir = os.path.join(args.out, job['name'])
    os.makedirs(run_dir, exist_ok=True)
    if is_done(run_dir, args.episodes):
        return job['name'], 'skipped', 0.0
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads), PPO_DEVICE=args.device)
    cmd = [sys.executable, os.path.join(HERE, job['script']),
           '--num_episodes', str(args.episodes), '--eval_every', str(args.eval_every),
           '--num_eval', str(args.num_eval), '--seed', str(job['seed']), '--save_dir', run_dir,
           '--reward_type', job['reward']]
    if 'enc_scenario' in job:
        cmd += ['--scenario', job['enc_scenario'], '--dcpa_band', job['band'], '--scale', args.scale,
                '--control', args.control]
        if args.trace:
            cmd += ['--trace', args.trace]
    elif 'scenario' in job:
        cmd += ['--scenario', job['scenario'], '--automation', job['automation']]
        if args.trace:
            cmd += ['--trace', args.trace, '--trace_scale', str(args.trace_scale)]
        if args.attack_range > 0:
            cmd += ['--attack_range', str(args.attack_range)]
    t0 = time.time()
    with open(os.path.join(run_dir, 'stdout.log'), 'w', encoding='utf-8', errors='replace') as log:
        rc = subprocess.call(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, env=env)
    wall = time.time() - t0
    with open(os.path.join(run_dir, 'stdout.log'), 'a', encoding='utf-8') as log:
        log.write('\n%s exit %d wall %.0f s\n' % (job['name'], rc, wall))
    return job['name'], 'ok' if rc == 0 else 'FAILED rc=%d' % rc, wall


# ------------------------------------------------------------------ summary
def read_eval(run_dir):
    path = os.path.join(run_dir, 'eval_logs', 'eval_result.txt')
    rows = []
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            next(f, None)
            for line in f:
                p = line.split()
                if len(p) >= 4:
                    rows.append([float(v) for v in p[:4]])
    return np.asarray(rows) if rows else np.zeros((0, 4))


def read_train(run_dir):
    path = os.path.join(run_dir, 'train_logs', 'train_result.txt')
    ret, ln, succ = [], [], []
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            next(f, None)
            for line in f:
                p = line.split()
                if len(p) >= 4:
                    ret.append(float(p[1])); ln.append(float(p[2])); succ.append(p[3] == 'True')
    return np.asarray(ret), np.asarray(ln), np.asarray(succ, dtype=float)


def summarise(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    runs = []
    for args_path in sorted(glob.glob(os.path.join(out, '*', 'args.json'))):
        run_dir = os.path.dirname(args_path)
        with open(args_path, encoding='utf-8') as f:
            a = json.load(f)
        ev = read_eval(run_dir)
        ret, ln, succ = read_train(run_dir)
        if 'dcpa_band' in a:
            task = 'enc'
            key = (task, a.get('scenario') + (' full' if a.get('control') == 'full' else ''), a.get('dcpa_band'),
                   family(a.get('reward_type')))
        elif 'scenario' in a:
            task = 'scen'
            key = (task, a.get('scenario'), a.get('automation'), family(a.get('reward_type')))
        else:                       # a run of the obstacle-scene trainers removed on 2026-09-24: not summarised
            continue
        reach = ev[ev[:, 3] >= 1.0, 0] if len(ev) else np.zeros(0)
        reach80 = ev[ev[:, 3] >= 0.8, 0] if len(ev) else np.zeros(0)
        runs.append(dict(run=os.path.basename(run_dir), task=task, obst_id=key[1], dest_id=key[2], reward=key[3],
                         seed=a.get('seed'), episodes=int(len(ret)),
                         first_eval_success_1=int(reach[0]) if len(reach) else '',
                         first_eval_success_08=int(reach80[0]) if len(reach80) else '',
                         final_eval_success=float(ev[-1, 3]) if len(ev) else '',
                         final_eval_length=float(ev[-1, 2]) if len(ev) else '',
                         final_eval_return=float(ev[-1, 1]) if len(ev) else '',
                         mean_eval_success=float(ev[:, 3].mean()) if len(ev) else '',
                         train_success_last500=float(succ[-500:].mean()) if len(succ) else '',
                         train_length_last500=float(ln[-500:].mean()) if len(ln) else '',
                         _ev=ev, _key=key))
    if not runs:
        print('no runs under', out)
        return
    cols = [c for c in runs[0] if not c.startswith('_')]
    with open(os.path.join(out, 'matrix_runs.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in runs:
            w.writerow({c: r[c] for c in cols})

    groups = {}
    for r in runs:
        groups.setdefault(r['_key'], []).append(r)

    def ms(vals):
        vals = [float(v) for v in vals if v != '']
        if not vals:
            return '', ''
        return round(float(np.mean(vals)), 2), round(float(np.std(vals)), 2)

    summary = []
    for key in sorted(groups):
        g = groups[key]
        row = dict(task=key[0], obst_id=key[1], dest_id=key[2], reward=key[3], n_seeds=len(g))
        for m in ('first_eval_success_1', 'first_eval_success_08', 'final_eval_success', 'final_eval_length',
                  'mean_eval_success', 'train_length_last500'):
            row[m + '_mean'], row[m + '_sd'] = ms([r[m] for r in g])
        row['seeds_reaching_1'] = sum(1 for r in g if r['first_eval_success_1'] != '')
        summary.append(row)
    with open(os.path.join(out, 'matrix_summary.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)

    scen = sorted({(k[0], k[1], k[2]) for k in groups})
    for col_idx, title, fname in ((3, 'evaluation success rate', 'matrix_success.png'),
                                  (2, 'evaluation mean length (decisions)', 'matrix_length.png')):
        n = len(scen)
        ncols = 2 if n > 4 else 1
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 3.4 * nrows), squeeze=False)
        for ax, (task, obst, dest) in zip(axes.flat, scen):
            for fam in ('sparse', 'dense'):
                g = groups.get((task, obst, dest, fam), [])
                curves = [r['_ev'] for r in g if len(r['_ev'])]
                if not curves:
                    continue
                for ev in curves:
                    ax.plot(ev[:, 0], ev[:, col_idx], '-', lw=0.8, alpha=0.4, color=REWARD_COLOUR[fam])
                m = min(len(ev) for ev in curves)
                mean = np.mean([ev[:m, col_idx] for ev in curves], axis=0)
                ax.plot(curves[0][:m, 0], mean, '-', lw=2.2, color=REWARD_COLOUR[fam],
                        label='%s (n=%d)' % (fam, len(curves)))
            ax.set_title('%s | %s / %s' % ({'scen': 'scenario attack', 'enc': 'encounter attack'}[task], obst, dest), fontsize=10)
            ax.set_xlabel('training episode')
            if col_idx == 3:
                ax.set_ylim(-0.02, 1.02)
            ax.legend(fontsize=8)
        for ax in list(axes.flat)[len(scen):]:
            ax.axis('off')
        fig.suptitle('%s (thin = seeds, thick = mean)' % title)
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), dpi=110)
        plt.close(fig)
    print('summary written to', out)
    for row in summary:
        print('%-4s %-22s %-5s %-6s n=%d  first@1.0 %s+-%s  final succ %s  final len %s+-%s' % (
            row['task'], row['obst_id'], row['dest_id'], row['reward'], row['n_seeds'],
            row['first_eval_success_1_mean'], row['first_eval_success_1_sd'], row['final_eval_success_mean'],
            row['final_eval_length_mean'], row['final_eval_length_sd']))


# --------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default='runs/matrix')
    p.add_argument('--episodes', type=int, default=6000)
    p.add_argument('--eval_every', type=int, default=500)
    p.add_argument('--num_eval', type=int, default=20)
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    p.add_argument('--tasks', nargs='*', choices=['scen', 'enc'], default=['enc'],
                   help='enc = main_attack_ppo_enc.py (default), scen = main_attack_ppo_scen.py')
    p.add_argument('--enc_scenarios', nargs='*', default=None, help='enc task: scenario specs (default mix)')
    p.add_argument('--bands', nargs='*', default=['passing'], help='enc task: initial DCPA bands')
    p.add_argument('--scale', default='arena', help='enc task: collision-standard scale')
    p.add_argument('--control', default='full', choices=['full', 'turn'], help='enc task: attacker control')
    p.add_argument('--scenarios', nargs='*', default=None, help='scen task: families (default all four)')
    p.add_argument('--automation', nargs='*', default=['fixed'], help='scen task: target automation levels (fixed = route playback)')
    p.add_argument('--trace', default='', help='scen task: trace .npy for data-shaped routes')
    p.add_argument('--trace_scale', type=float, default=1.0, help='scen task: length factor for the trace shapes (1.0 = full size)')
    p.add_argument('--attack_range', type=float, default=0.0, help='scen task: success distance (0 = one target length)')
    p.add_argument('--parallel', type=int, default=8)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--device', default='cpu')
    p.add_argument('--summary-only', action='store_true')
    args = p.parse_args()
    args.out = os.path.abspath(os.path.join(HERE, args.out)) if not os.path.isabs(args.out) else args.out
    os.makedirs(args.out, exist_ok=True)
    if not args.summary_only:
        jobs = make_jobs(args)
        print('%d jobs, %d parallel, %d threads each, device %s, %d episodes' % (
            len(jobs), args.parallel, args.threads, args.device, args.episodes), flush=True)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = [pool.submit(run_job, j, args) for j in jobs]
            for k, fut in enumerate(as_completed(futures), 1):
                name, status, wall = fut.result()
                print('[%2d/%d] %-36s %-8s %6.0f s   (elapsed %.0f s)' % (k, len(jobs), name, status, wall,
                                                                          time.time() - t0), flush=True)
        print('matrix finished in %.0f s' % (time.time() - t0), flush=True)
    summarise(args.out)


if __name__ == '__main__':
    main()
