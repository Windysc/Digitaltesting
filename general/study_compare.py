"""
study_compare.py -- the method comparison of review/STUDY_PLAN_PEER_REVIEW.md (sections 3-5), 2026-10-08.

Every method attacks the SAME fixed list of initial encounters (scenario name, encounter seed), drawn
from the held-out TEST parameter intervals (attack_scenarios.PARAM_SPLITS), against one system under
test (the target's collision-avoidance preset, --sut), under the same attacker limits and 6 s
decisions. Per encounter a method may spend up to --budget episodes; it stops at the first failure
it creates (domain violation held, attack_scenarios success) and the attempt index is logged, so the
budget curve (fraction of encounters with a failure found vs episodes per encounter) follows directly.

Methods
  hold | intercept | pursuit   scripted closed-loop references (one deterministic episode)
  random                       uniform random 3-segment open-loop manoeuvre (action level, duration) x 3
  cem                          cross-entropy method over the same 3-segment manoeuvre
  ppo                          a trained checkpoint: attempt 1 = argmax policy, then sampled episodes

Sub-commands
  run     one (method, sut, seed) -> <out>/<sut>/<method>_s<seed>.csv (one row per encounter)
  report  all csv under <out> -> summary.csv, summary.md, budget_curves.png

Simulator cost is logged in episodes, transitions (decisions) and 1 s sub-steps; the PPO training cost
is read from the run's eval_result.txt (transitions and sub-steps up to the selected checkpoint).
"""
import argparse
import csv
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_moving_obj import ownship                                   # noqa: E402
from attack_scenarios import EncounterAttackEnv, BASELINES, resolve_scenarios   # noqa: E402

N_SEG = 3


# ------------------------------------------------------------------ problem
def encounter_list(scenario='mix', per_scenario=6, base_seed=20000):
    names = resolve_scenarios(scenario)
    return [(n, base_seed + 100 * j + i) for j, n in enumerate(names) for i in range(per_scenario)]


def make_env(sut, encounters, v_max=6.0, split='test', band='passing', save_dir='.'):
    return EncounterAttackEnv(ownship(0, 0, 0, 0, 0, 0), scenario='mix', dcpa_band=band, automation=sut,
                              param_split=split, encounter_list=encounters, v_range=(3.0, v_max),
                              save_dir=save_dir, seed=0)


def reset_to(env, i):
    env._enc_i = i
    return env.reset()


def rollout(env, i, policy):
    """One episode on encounter i; policy(env, obs, k) -> action. Returns the evaluation dict."""
    obs = reset_to(env, i)
    done, k = False, 0
    while not done:
        obs, _, done, _ = env.step(policy(env, obs, k))
        k += 1
    return env.evaluation()


# ------------------------------------------------------------------ open-loop manoeuvre
def plan_policy(plan):
    """plan = [(action, duration), ...] then straight at constant speed (the zero action)."""
    seq = [a for a, d in plan for _ in range(int(d))]

    def pol(env, obs, k):
        return seq[k] if k < len(seq) else env.zero_action
    return pol


def zero_action(env):
    return int(np.argmin([abs(a) + abs(r) for a, r in env.action_table]))


class RandomSearch:
    def __init__(self, env, rng, d_max):
        self.env, self.rng, self.d_max = env, rng, d_max

    def propose(self):
        n = len(self.env.action_table)
        return [(int(self.rng.randint(n)), int(self.rng.randint(1, self.d_max + 1))) for _ in range(N_SEG)]

    def tell(self, plan, score):
        pass


class CEM:
    """Cross-entropy method: per segment a categorical over the actions and a normal over the duration."""

    def __init__(self, env, rng, d_max, pop=16, elite=4, smooth=0.7):
        self.env, self.rng, self.d_max = env, rng, d_max
        self.n = len(env.action_table)
        self.p = np.full((N_SEG, self.n), 1.0 / self.n)
        self.mu = np.full(N_SEG, d_max / 2.0)
        self.sd = np.full(N_SEG, d_max / 3.0)
        self.pop, self.elite, self.smooth = pop, elite, smooth
        self.batch = []

    def propose(self):
        return [(int(self.rng.choice(self.n, p=self.p[s])),
                 int(np.clip(round(self.rng.normal(self.mu[s], self.sd[s])), 1, self.d_max))) for s in range(N_SEG)]

    def tell(self, plan, score):
        self.batch.append((score, plan))
        if len(self.batch) < self.pop:
            return
        best = [p for _, p in sorted(self.batch, key=lambda t: -t[0])[:self.elite]]
        for s in range(N_SEG):
            freq = np.bincount([p[s][0] for p in best], minlength=self.n) / len(best)
            self.p[s] = self.smooth * freq + (1 - self.smooth) * self.p[s]
            self.p[s] = np.maximum(self.p[s], 1e-3); self.p[s] /= self.p[s].sum()
            d = np.array([p[s][1] for p in best], float)
            self.mu[s] = self.smooth * d.mean() + (1 - self.smooth) * self.mu[s]
            self.sd[s] = max(1.0, self.smooth * d.std() + (1 - self.smooth) * self.sd[s])
        self.batch = []


def score(ev, std):
    """Search objective: a created failure first, else the smallest miss distance (worst margin)."""
    return 1e4 * ev['success'] + 1e3 * ev['collision'] - ev['min_distance'] / max(std.d_safe, 1.0)


# ------------------------------------------------------------------ PPO deployment
def load_ppo(run_dir, env):
    from PPO import PPO
    mdir = os.path.join(run_dir, 'models', 'MassTestingEnv')
    cands = sorted(glob.glob(os.path.join(mdir, '*_best.pth'))) or sorted(glob.glob(os.path.join(mdir, '*_final.pth'))) \
        or sorted(glob.glob(os.path.join(mdir, 'PPO_MassTestingEnv_seed*[0-9].pth')))
    if not cands:
        raise FileNotFoundError('no checkpoint in %s' % mdir)
    agent = PPO(env.observation_space.shape[0], env.action_space.n, 3e-4, 1e-3, 0.99, 1, 0.2, False, 0.6)
    agent.load(cands[0])
    return agent, cands[0]


def ppo_policy(agent, deterministic):
    def pol(env, obs, k):
        return agent.select_action(obs, deterministic=deterministic)
    return pol


def training_cost(run_dir, ckpt):
    """(episodes, transitions, sub-steps) used by PPO up to the selected checkpoint (incl. selection evals)."""
    path = os.path.join(run_dir, 'eval_logs', 'eval_result.txt')
    rows = [l.split() for l in open(path).read().splitlines()[1:] if l.strip()]
    best_ep, best = None, -1.0
    for r in rows:
        if float(r[3]) > best:
            best, best_ep = float(r[3]), r
    r = best_ep if (best_ep is not None and ckpt.endswith('_best.pth')) else rows[-1]
    n_eval = json.load(open(os.path.join(run_dir, 'args.json')))['num_eval']
    i_ep = int(r[0])
    k = sum(1 for x in rows if int(x[0]) <= i_ep)
    return i_ep + k * n_eval, int(r[4]), int(r[5])


# ------------------------------------------------------------------ run
def run(args):
    encs = encounter_list(args.scenario, args.per_scenario, args.enc_seed)
    env = make_env(args.sut, encs, v_max=args.v_max, split=args.split, save_dir=args.tmp)
    env.zero_action = zero_action(env)
    rng = np.random.RandomState(args.seed)
    d_max = max(1, env.cap_decisions // N_SEG)
    out_dir = os.path.join(args.out, args.sut)
    os.makedirs(out_dir, exist_ok=True)
    tag = args.method if args.method != 'ppo' else 'ppo' + (args.ppo_tag or '')
    rows, extra = [], {}
    agent = None
    if args.method == 'ppo':
        agent, ckpt = load_ppo(args.run_dir, env)
        extra = dict(zip(('train_episodes', 'train_transitions', 'train_substeps'), training_cost(args.run_dir, ckpt)))
        extra['checkpoint'] = os.path.relpath(ckpt, args.run_dir)
        import torch
        torch.manual_seed(args.seed)
    for i in range(len(encs)):
        dec0, sub0 = env.sim_decisions, env.sim_substeps
        first, coll, best_margin, n_try = -1, 0, math.inf, 0
        evasions = 0
        if args.method in BASELINES:
            fn = BASELINES[args.method]
            budget = 1
        else:
            budget = args.budget
            searcher = (RandomSearch(env, rng, d_max) if args.method == 'random' else
                        CEM(env, rng, d_max) if args.method == 'cem' else None)
        for k in range(budget):
            if args.method in BASELINES:
                ev = rollout(env, i, lambda e, o, kk: fn(e, o))
            elif args.method == 'ppo':
                ev = rollout(env, i, ppo_policy(agent, deterministic=(k == 0)))
            else:
                plan = searcher.propose()
                ev = rollout(env, i, plan_policy(plan))
                searcher.tell(plan, score(ev, env.std))
            n_try += 1
            evasions = max(evasions, ev['target_evasions'])
            best_margin = min(best_margin, ev['min_distance'])
            if ev['success']:
                first, coll = k + 1, ev['collision']
                break
        rows.append(dict(sut=args.sut, method=tag, seed=args.seed, encounter=i, scenario=encs[i][0],
                         family=ev['family'], encounter_seed=encs[i][1], found=int(first > 0), first_attempt=first,
                         collision=coll, attempts=n_try, transitions=env.sim_decisions - dec0,
                         substeps=env.sim_substeps - sub0, min_distance=round(best_margin, 1),
                         max_target_evasions=evasions, **extra))
    path = os.path.join(out_dir, '%s_s%d.csv' % (tag, args.seed))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    found = np.mean([r['found'] for r in rows])
    print('%s %s seed %d: found %.3f | transitions %d' % (args.sut, tag, args.seed, found,
                                                         sum(r['transitions'] for r in rows)))


# ------------------------------------------------------------------ report
def wilson(k, n, z=1.96):
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def report(args):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    files = glob.glob(os.path.join(args.out, '*', '*.csv'))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    B = args.budget
    lines, summ = [], []
    for (sut, m), g in df.groupby(['sut', 'method']):
        per_seed = g.groupby('seed')['found'].mean()
        k, n = int(g['found'].sum()), len(g)
        lo, hi = wilson(k, n)
        cost = g.groupby('seed')[['transitions', 'substeps']].sum().mean()
        found_at = {b: g.assign(f=(g['found'] == 1) & (g['first_attempt'] <= b)).groupby('seed')['f'].mean().mean()
                    for b in (1, 10, 100) if b <= B}
        row = dict(sut=sut, method=m, seeds=per_seed.size, encounters=g['encounter'].nunique(),
                   found=round(k / n, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                   seed_sd=round(per_seed.std(ddof=1), 3) if per_seed.size > 1 else 0.0,
                   found_at_1=round(found_at.get(1, math.nan), 3), found_at_10=round(found_at.get(10, math.nan), 3),
                   collision_given_found=round(g.loc[g['found'] == 1, 'collision'].mean(), 3) if k else math.nan,
                   median_min_distance_not_found=round(g.loc[g['found'] == 0, 'min_distance'].median(), 1) if k < n else math.nan,
                   search_transitions_per_seed=int(cost['transitions']), search_substeps_per_seed=int(cost['substeps']))
        if 'train_transitions' in g and g['train_transitions'].notna().any():
            row['train_transitions_per_seed'] = int(g.groupby('seed')['train_transitions'].first().mean())
        for fam, gf in g.groupby('family'):
            row['found_' + fam] = round(gf['found'].mean(), 3)
        summ.append(row)
    S = pd.DataFrame(summ)
    S.to_csv(os.path.join(args.out, 'summary.csv'), index=False)
    # paired bootstrap: method difference on the common (seed, encounter) grid, resampling encounters and seeds
    rng = np.random.RandomState(0)
    pairs = []
    for sut, g in df.groupby('sut'):
        piv = g.pivot_table(index=['encounter'], columns='method', values='found', aggfunc='mean')
        ms = list(piv.columns)
        for a in ms:
            for b in ms:
                if a >= b:
                    continue
                d = (piv[a] - piv[b]).values
                bs = [d[rng.randint(len(d), size=len(d))].mean() for _ in range(2000)]
                pairs.append(dict(sut=sut, a=a, b=b, diff=round(d.mean(), 3),
                                  ci_lo=round(np.percentile(bs, 2.5), 3), ci_hi=round(np.percentile(bs, 97.5), 3)))
    P = pd.DataFrame(pairs)
    P.to_csv(os.path.join(args.out, 'paired_bootstrap.csv'), index=False)
    with open(os.path.join(args.out, 'summary.md'), 'w') as f:
        f.write(S.to_markdown(index=False) + '\n\n' + P.to_markdown(index=False) + '\n')
    # budget curves
    suts = sorted(df['sut'].unique())
    fig, axs = plt.subplots(1, len(suts), figsize=(4.2 * len(suts), 3.6), sharey=True, squeeze=False)
    bs_ = np.arange(1, B + 1)
    for ax, sut in zip(axs[0], suts):
        for m, g in df[df['sut'] == sut].groupby('method'):
            curve = [((g['found'] == 1) & (g['first_attempt'] <= b)).groupby(g['seed']).mean().mean() for b in bs_]
            ax.plot(bs_, curve, label=m)
        ax.set_xscale('log'); ax.set_title('SUT: %s' % sut); ax.set_xlabel('episodes per encounter')
        ax.grid(alpha=.3)
    axs[0][0].set_ylabel('encounters with a failure found'); axs[0][-1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, 'budget_curves.png'), dpi=130)
    print(S.to_string(index=False))
    print(P.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run', 'report'])
    ap.add_argument('--method', default='random')
    ap.add_argument('--sut', default='replan')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--budget', type=int, default=100, help='episodes per encounter (search methods, PPO retries)')
    ap.add_argument('--scenario', default='mix')
    ap.add_argument('--per_scenario', type=int, default=6)
    ap.add_argument('--enc_seed', type=int, default=20000)
    ap.add_argument('--split', default='test')
    ap.add_argument('--v_max', type=float, default=6.0)
    ap.add_argument('--run_dir', default='')
    ap.add_argument('--ppo_tag', default='')
    ap.add_argument('--out', default='runs/study')
    ap.add_argument('--tmp', default='.')
    args = ap.parse_args()
    run(args) if args.cmd == 'run' else report(args)


if __name__ == '__main__':
    main()
