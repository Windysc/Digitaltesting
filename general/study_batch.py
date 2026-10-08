"""
study_batch.py -- runs the study of review/STUDY_PLAN_PEER_REVIEW.md as a process pool, 2026-10-08.

  python study_batch.py train   --root <dir>   PPO adversaries: SUT x seed (+ the Monte-Carlo ablation)
  python study_batch.py compare --root <dir>   every method on the common held-out encounters
  python study_batch.py report  --root <dir>
  python study_batch.py search  --root <dir> --sut replan --methods random,cem --budget 300
                                               the search methods at another per-encounter budget,
                                               into compare_b<budget>/ (equal-total-budget check)

Layout under <root>: ppo/<sut>_<adv>_s<seed>/ (training runs), compare/<sut>/<method>_s<seed>.csv,
logs/. Jobs whose output exists are skipped, so a stopped batch resumes.
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SUTS = ('fixed', 'manual', 'autonomous', 'replan')
SEEDS = range(1, 11)
ABLATION_SUTS = ('autonomous', 'replan')         # PPO with the reference Monte-Carlo update as well


def train_jobs(a):
    jobs = []
    for adv, suts in (('gae', SUTS), ('mc', ABLATION_SUTS)):
        for sut in suts:
            for s in SEEDS:
                d = os.path.join(a.root, 'ppo', '%s_%s_s%d' % (sut, adv, s))
                if os.path.exists(os.path.join(d, 'done.txt')):
                    continue
                cmd = [sys.executable, os.path.join(HERE, 'main_attack_ppo_enc.py'), '--save_dir', d, '--seed', str(s),
                       '--num_episodes', str(a.episodes), '--eval_every', str(a.eval_every), '--num_eval', '48',
                       '--automation', sut, '--param_split', 'train', '--select_seed', str(5000 + s),
                       '--advantage', adv, '--eval_deterministic', '1', '--speed_range', '3,%g' % a.v_max,
                       '--reward_type', 'dense_attack_reward']
                jobs.append(('%s_%s_s%d' % (sut, adv, s), cmd, os.path.join(d, 'done.txt')))
    return jobs


def compare_jobs(a):
    jobs = []
    out = os.path.join(a.root, a.compare_dir)
    base = [sys.executable, os.path.join(HERE, 'study_compare.py'), 'run', '--out', out, '--budget', str(a.budget),
            '--v_max', '%g' % a.v_max, '--tmp', os.path.join(a.root, 'logs')]
    for sut in SUTS:
        for m in ('hold', 'intercept', 'pursuit'):
            jobs.append(('%s_%s' % (sut, m), base + ['--method', m, '--sut', sut, '--seed', '1'],
                         os.path.join(out, sut, '%s_s1.csv' % m)))
        for m in ('random', 'cem'):
            for s in SEEDS:
                jobs.append(('%s_%s_s%d' % (sut, m, s), base + ['--method', m, '--sut', sut, '--seed', str(s)],
                             os.path.join(out, sut, '%s_s%d.csv' % (m, s))))
        for adv in ('gae', 'mc'):
            if adv == 'mc' and sut not in ABLATION_SUTS:
                continue
            for s in SEEDS:
                rd = os.path.join(a.root, 'ppo', '%s_%s_s%d' % (sut, adv, s))
                tag = '_' + adv
                jobs.append(('%s_ppo%s_s%d' % (sut, tag, s),
                             base + ['--method', 'ppo', '--run_dir', rd, '--ppo_tag', tag, '--sut', sut, '--seed', str(s)],
                             os.path.join(out, sut, 'ppo%s_s%d.csv' % (tag, s))))
    return [j for j in jobs if not os.path.exists(j[2])]


def search_jobs(a):
    out = os.path.join(a.root, 'compare_b%d' % a.budget)
    base = [sys.executable, os.path.join(HERE, 'study_compare.py'), 'run', '--out', out, '--budget', str(a.budget),
            '--v_max', '%g' % a.v_max, '--tmp', os.path.join(a.root, 'logs')]
    jobs = []
    for sut in a.sut.split(','):
        for m in a.methods.split(','):
            for s in SEEDS:
                jobs.append(('b%d_%s_%s_s%d' % (a.budget, sut, m, s), base + ['--method', m, '--sut', sut, '--seed', str(s)],
                             os.path.join(out, sut, '%s_s%d.csv' % (m, s))))
    return [j for j in jobs if not os.path.exists(j[2])]


def run_pool(jobs, workers, logdir, mark_done):
    os.makedirs(logdir, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    t0 = time.time()

    def one(job):
        name, cmd, done = job
        work = os.path.join(logdir, 'work')
        os.makedirs(work, exist_ok=True)
        with open(os.path.join(logdir, name + '.log'), 'w') as log:
            t = time.time()
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=work)
        if rc == 0 and mark_done:
            with open(done, 'w') as f:
                f.write('%.0f s\n' % (time.time() - t))
        print('%-28s rc=%d %6.0f s  (elapsed %.0f min)' % (name, rc, time.time() - t, (time.time() - t0) / 60), flush=True)
        return rc

    with ThreadPoolExecutor(workers) as ex:
        rcs = list(ex.map(one, jobs))
    print('finished %d jobs, %d failed, %.1f min' % (len(rcs), sum(r != 0 for r in rcs), (time.time() - t0) / 60))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['train', 'compare', 'report', 'search'])
    ap.add_argument('--sut', default='replan', help='search: comma list of systems under test')
    ap.add_argument('--methods', default='random,cem', help='search: comma list of search methods')
    ap.add_argument('--root', required=True)
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--episodes', type=int, default=3000)
    ap.add_argument('--eval_every', type=int, default=250)
    ap.add_argument('--budget', type=int, default=100)
    ap.add_argument('--v_max', type=float, default=6.0)
    ap.add_argument('--compare_dir', default='compare', help='compare / report: result folder under --root')
    a = ap.parse_args()
    a.root = os.path.abspath(a.root)
    if a.cmd == 'train':
        jobs = train_jobs(a)
        print('%d training jobs' % len(jobs), flush=True)
        run_pool(jobs, a.workers, os.path.join(a.root, 'logs'), True)
    elif a.cmd == 'search':
        jobs = search_jobs(a)
        print('%d search jobs at %d episodes per encounter' % (len(jobs), a.budget), flush=True)
        run_pool(jobs, a.workers, os.path.join(a.root, 'logs'), False)
    elif a.cmd == 'compare':
        jobs = compare_jobs(a)
        print('%d comparison jobs' % len(jobs), flush=True)
        run_pool(jobs, a.workers, os.path.join(a.root, 'logs'), False)
    else:
        # extra-budget result folders only (compare_b<budget>), never the batch logs next to them
        extra = sorted(d for d in glob.glob(os.path.join(a.root, 'compare_b*'))
                       if os.path.isdir(d) and re.fullmatch(r'compare_b\d+', os.path.basename(d)))
        sys.exit(subprocess.call([sys.executable, os.path.join(HERE, 'study_compare.py'), 'report', '--out',
                                  os.path.join(a.root, a.compare_dir), '--budget', str(a.budget),
                                  '--ppo_root', os.path.join(a.root, 'ppo'), '--extra', ','.join(extra)]))


if __name__ == '__main__':
    main()
