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
  run       one (method, sut, seed) -> <out>/<sut>/<method>_s<seed>.csv (one row per encounter)
  report    all csv under <out> -> summary.csv, summary.md, paired_bootstrap.csv, budget_curves.png
  envelope  scripted attackers x SUT x attacker top speed on the same encounters -> <out>/envelope.csv
            (where the benchmark saturates: --v_list 6,7.5,9,12 = speed ratio 1.0-2.0 to the 6 m/s target)

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


def training_cost(run_dir, ckpt=None):
    """Simulator cost of obtaining the deployed PPO policy: the WHOLE training run (the best checkpoint is
    only known once it ends) plus every checkpoint-selection evaluation (separate encounters, counted from
    the per-episode eval_dicts files). Conservative for PPO."""
    import pandas as pd
    path = os.path.join(run_dir, 'eval_logs', 'eval_result.txt')
    rows = [l.split() for l in open(path).read().splitlines()[1:] if l.strip()]
    best_ep, best = int(rows[-1][0]), 0.0
    for r in rows:                                   # the trainer saves _best on a strictly higher success rate
        if float(r[3]) > best:
            best, best_ep = float(r[3]), int(r[0])
    sel = [pd.read_csv(f) for f in glob.glob(os.path.join(run_dir, 'eval_logs', 'eval_dicts_episode_*.csv'))]
    sel = pd.concat(sel, ignore_index=True) if sel else pd.DataFrame(columns=['steps', 'substeps'])
    return dict(train_episodes=int(rows[-1][0]), train_transitions=int(rows[-1][4]), train_substeps=int(rows[-1][5]),
                select_episodes=len(sel), select_transitions=int(sel['steps'].sum()),
                select_substeps=int(sel['substeps'].sum()) if 'substeps' in sel else -1,
                selected_episode=best_ep if (ckpt is None or str(ckpt).endswith('_best.pth')) else int(rows[-1][0]))


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
        extra = training_cost(args.run_dir, ckpt)
        extra['checkpoint'] = os.path.relpath(ckpt, args.run_dir)
        import torch
        torch.manual_seed(args.seed)
    events = []
    for i in range(len(encs)):
        dec0, sub0 = env.sim_decisions, env.sim_substeps
        first, coll, best_margin, n_try = -1, 0, math.inf, 0
        evasions = 0
        kept = None                                    # actions of the event, else of the closest approach
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
                agent.buffer.clear()          # sampled actions are stored for training; deployment never trains
            else:
                plan = searcher.propose()
                ev = rollout(env, i, plan_policy(plan))
                searcher.tell(plan, score(ev, env.std))
            n_try += 1
            evasions = max(evasions, ev['target_evasions'])
            if ev['min_distance'] < best_margin or ev['success']:
                kept = (k + 1, [int(a) for a in env.ownship_action])
            best_margin = min(best_margin, ev['min_distance'])
            if ev['success']:
                first, coll = k + 1, ev['collision']
                break
        # every action sequence replays exactly (deterministic environment): event_labels.py marks the events from it
        events.append(dict(sut=args.sut, method=tag, seed=args.seed, encounter=i, scenario=encs[i][0],
                           encounter_seed=encs[i][1], kind='event' if first > 0 else 'closest', attempt=kept[0],
                           v_max=args.v_max, split=args.split, scenario_spec=args.scenario,
                           per_scenario=args.per_scenario, enc_seed=args.enc_seed, actions=kept[1]))
        rows.append(dict(sut=args.sut, method=tag, seed=args.seed, encounter=i, scenario=encs[i][0],
                         family=ev['family'], encounter_seed=encs[i][1], found=int(first > 0), first_attempt=first,
                         collision=coll, attempts=n_try, transitions=env.sim_decisions - dec0,
                         substeps=env.sim_substeps - sub0, min_distance=round(best_margin, 1),
                         max_target_evasions=evasions, **extra))
    path = os.path.join(out_dir, '%s_s%d.csv' % (tag, args.seed))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(out_dir, '%s_s%d.events.jsonl' % (tag, args.seed)), 'w') as f:
        for e in events:
            f.write(json.dumps(e) + '\n')
    found = np.mean([r['found'] for r in rows])
    print('%s %s seed %d: found %.3f | transitions %d' % (args.sut, tag, args.seed, found,
                                                         sum(r['transitions'] for r in rows)))


# ------------------------------------------------------------------ envelope
def envelope(args):
    import pandas as pd
    encs = encounter_list(args.scenario, args.per_scenario, args.enc_seed)
    rows = []
    for v in [float(x) for x in args.v_list.split(',')]:
        for sut in args.suts.split(','):
            env = make_env(sut, encs, v_max=v, split=args.split, save_dir=args.tmp)
            for m in ('hold', 'intercept', 'pursuit'):
                fn = BASELINES[m]
                evs = [rollout(env, i, lambda e, o, k: fn(e, o)) for i in range(len(encs))]
                d = pd.DataFrame(evs)
                row = dict(v_max=v, speed_ratio=round(v / env.cruise, 2), sut=sut, method=m, encounters=len(d),
                           found=round(d['success'].mean(), 3), collision=round(d['collision'].mean(), 3),
                           clear=round((d['outcome'] == 'clear').mean(), 3),
                           unresolved=round((d['outcome'] == 'unresolved').mean(), 3),
                           off_map=round((d['outcome'] == 'off_map').mean(), 3),
                           target_evasions=round(d['target_evasions'].mean(), 2), mean_steps=round(d['steps'].mean(), 1))
                row.update({'found_' + f: round(g['success'].mean(), 2) for f, g in d.groupby('family')})
                rows.append(row)
                print(row, flush=True)
    os.makedirs(args.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, 'envelope.csv'), index=False)


# ------------------------------------------------------------------ report
def wilson(k, n, z=1.96):
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


SUT_ORDER = ['fixed', 'manual', 'autonomous', 'replan']
SUT_TITLE = {'fixed': 'fixed track (sanity case)', 'manual': 'manual (one alteration, 20 s latency)',
             'autonomous': 'autonomous (one alteration)', 'replan': 'replan (re-planning every 6 s)'}
LEGEND_ORDER = ['hold course', 'pursuit', 'intercept', 'random search, 100 ep', 'random search, 300 ep',
                'CEM, 100 ep', 'CEM, 300 ep', 'PPO (Monte-Carlo update)', 'PPO (GAE)']


def method_style(m):
    """Fixed plotting style per method, the same in every panel."""
    table = {'hold': dict(color='0.55', marker='x', ls='none', label='hold course'),
             'pursuit': dict(color='0.35', marker='s', ls='none', label='pursuit'),
             'intercept': dict(color='black', marker='o', ls='none', label='intercept'),
             'random': dict(color='tab:blue', ls='--', lw=1.3, label='random search, 100 ep'),
             'random_b300': dict(color='tab:blue', ls='-', lw=1.6, label='random search, 300 ep'),
             'cem': dict(color='tab:orange', ls='--', lw=1.3, label='CEM, 100 ep'),
             'cem_b300': dict(color='tab:orange', ls='-', lw=1.6, label='CEM, 300 ep'),
             'ppo_mc': dict(color='tab:red', ls='--', lw=1.3, label='PPO (Monte-Carlo update)'),
             'ppo_gae': dict(color='tab:red', ls='-', lw=1.8, label='PPO (GAE)')}
    return dict(table.get(m, dict(color='tab:green', ls='-', label=m)))


def _load(out, budget, label=''):
    import pandas as pd
    files = glob.glob(os.path.join(out, '*', '*.csv'))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['budget'] = np.where(df['method'].isin(list(BASELINES)), 1, budget)
    if label:
        df['method'] = df['method'] + label
    return df


def _matrix(g):
    """seeds x encounters array of the found flag (one method, one SUT)."""
    piv = g.pivot_table(index='seed', columns='encounter', values='found', aggfunc='mean')
    return piv.values, list(piv.columns)


def _boot(mats, rng, n=2000):
    """Two-level bootstrap: encounters resampled jointly for all methods, seeds resampled per method.
    mats: list of seeds x encounters arrays on the same encounter columns. Returns n x len(mats) means."""
    n_enc = mats[0].shape[1]
    out = np.empty((n, len(mats)))
    for b in range(n):
        e = rng.randint(n_enc, size=n_enc)
        for j, m in enumerate(mats):
            s = rng.randint(m.shape[0], size=m.shape[0])
            out[b, j] = m[np.ix_(s, e)].mean()
    return out


def _found_within(g, b):
    """seeds x encounters array: failure induced within b episodes per encounter."""
    f = ((g['found'] == 1) & (g['first_attempt'] <= b)).astype(float)
    return g.assign(f=f).pivot_table(index='seed', columns='encounter', values='f', aggfunc='mean').values


def _spend_within(g, b):
    """Mean total transitions per seed for a budget of b episodes per encounter (search part approximated by
    attempts x mean transitions per attempt of each encounter) plus PPO training and checkpoint selection."""
    per_try = g['transitions'] / g['attempts'].clip(lower=1)
    search = (np.minimum(g['attempts'], b) * per_try).groupby(g['seed']).sum()
    fixed = g.groupby('seed')[['train_transitions', 'select_transitions']].first().sum(axis=1)
    return float((search + fixed).mean())


def matched_budget(df, rng):
    """Every method at its own budget, PPO also read at 1 and 10 episodes per encounter, with costs; and the
    paired differences of each PPO reading against the scripted intercept and each search method at full budget."""
    import pandas as pd
    rows, pairs = [], []
    for sut, gs in df.groupby('sut'):
        entries = []
        for m, g in gs.groupby('method'):
            B = int(g['budget'].iloc[0])
            budgets = sorted({1, 10, B}) if m.startswith('ppo') else [B]
            for b in budgets:
                entries.append((m, b, g))
        mats = {}
        for m, b, g in entries:
            mat = _found_within(g, b)
            mats[(m, b)] = mat
            bs = _boot([mat], rng)[:, 0]
            rows.append(dict(sut=sut, method=m, episodes_per_encounter=b, seeds=mat.shape[0], found=round(mat.mean(), 3),
                             boot_lo=round(np.percentile(bs, 2.5), 3), boot_hi=round(np.percentile(bs, 97.5), 3),
                             total_transitions_per_seed=int(_spend_within(g, b))))
        refs = [(m, b) for (m, b, g) in entries if not m.startswith('ppo') and m not in ('hold', 'pursuit')]
        for (m, b) in [(m, b) for (m, b, g) in entries if m.startswith('ppo')]:
            for (r, rb) in refs:
                bs = _boot([mats[(m, b)], mats[(r, rb)]], rng)
                d = bs[:, 0] - bs[:, 1]
                pairs.append(dict(sut=sut, a='%s<=%d' % (m, b), b='%s<=%d' % (r, rb),
                                  diff=round(mats[(m, b)].mean() - mats[(r, rb)].mean(), 3),
                                  ci_lo=round(np.percentile(d, 2.5), 3), ci_hi=round(np.percentile(d, 97.5), 3)))
    return pd.DataFrame(rows), pd.DataFrame(pairs)


def report(args):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    parts = [_load(args.out, args.budget)]
    for extra in [x for x in args.extra.split(',') if x]:
        b = int(os.path.basename(os.path.normpath(extra)).split('_b')[-1])
        parts.append(_load(extra, b, label='_b%d' % b))
    df = pd.concat([p for p in parts if p is not None], ignore_index=True)
    ppo_cost = {}
    if args.ppo_root:
        for d in glob.glob(os.path.join(args.ppo_root, '*_s*')):
            sut, adv, sd = os.path.basename(d).rsplit('_', 2)
            if os.path.exists(os.path.join(d, 'eval_logs', 'eval_result.txt')):
                ppo_cost[(sut, 'ppo_' + adv, int(sd[1:]))] = training_cost(d)
    for c in ('train_transitions', 'select_transitions'):
        df[c] = [ppo_cost.get((r.sut, r.method, r.seed), {}).get(c, 0) for r in df.itertuples()]
    rng = np.random.RandomState(0)
    summ, pairs, mats = [], [], {}
    for (sut, m), g in df.groupby(['sut', 'method']):
        mat, cols = _matrix(g)
        mats[(sut, m)] = (mat, cols)
        bs = _boot([mat], rng)[:, 0]
        per_seed = mat.mean(axis=1)
        k, n = int(g['found'].sum()), len(g)
        spend = g.groupby('seed')[['transitions', 'substeps']].sum().mean()
        fixed_cost = g.groupby('seed')[['train_transitions', 'select_transitions']].first().mean()
        B = int(g['budget'].iloc[0])
        row = dict(sut=sut, method=m, seeds=mat.shape[0], encounters=mat.shape[1], budget=B,
                   found=round(mat.mean(), 3), boot_lo=round(np.percentile(bs, 2.5), 3),
                   boot_hi=round(np.percentile(bs, 97.5), 3),
                   wilson_lo=round(wilson(k, n)[0], 3) if mat.shape[0] == 1 else math.nan,
                   wilson_hi=round(wilson(k, n)[1], 3) if mat.shape[0] == 1 else math.nan,
                   seed_sd=round(per_seed.std(ddof=1), 3) if mat.shape[0] > 1 else 0.0)
        for b in (1, 10, 100, 300):
            if b <= B:
                row['found_at_%d' % b] = round(((g['found'] == 1) & (g['first_attempt'] <= b)).groupby(g['seed']).mean().mean(), 3)
        row.update(collision_given_found=round(g.loc[g['found'] == 1, 'collision'].mean(), 3) if k else math.nan,
                   median_min_distance_not_found=round(g.loc[g['found'] == 0, 'min_distance'].median(), 1) if k < n else math.nan,
                   search_transitions_per_seed=int(spend['transitions']),
                   train_transitions_per_seed=int(fixed_cost['train_transitions']),
                   select_transitions_per_seed=int(fixed_cost['select_transitions']),
                   total_transitions_per_seed=int(spend['transitions'] + fixed_cost.sum()))
        for fam, gf in g.groupby('family'):
            row['found_' + fam] = round(gf['found'].mean(), 3)
        summ.append(row)
    S = pd.DataFrame(summ)
    for sut in sorted(df['sut'].unique()):
        ms = sorted(m for (s_, m) in mats if s_ == sut)
        for i, a in enumerate(ms):
            for b in ms[i + 1:]:
                (ma, ca), (mb, cb) = mats[(sut, a)], mats[(sut, b)]
                if ca != cb:
                    continue
                bs = _boot([ma, mb], rng)
                d = bs[:, 0] - bs[:, 1]
                pairs.append(dict(sut=sut, a=a, b=b, diff=round(ma.mean() - mb.mean(), 3),
                                  ci_lo=round(np.percentile(d, 2.5), 3), ci_hi=round(np.percentile(d, 97.5), 3),
                                  p_le_0=round(float((d <= 0).mean()), 4)))
    P = pd.DataFrame(pairs)
    S.to_csv(os.path.join(args.out, 'summary.csv'), index=False)
    P.to_csv(os.path.join(args.out, 'paired_bootstrap.csv'), index=False)
    M, MP = matched_budget(df, rng)
    M.to_csv(os.path.join(args.out, 'matched_budget.csv'), index=False)
    MP.to_csv(os.path.join(args.out, 'matched_pairs.csv'), index=False)
    with open(os.path.join(args.out, 'summary.md'), 'w') as f:
        f.write(S.to_markdown(index=False) + '\n\n' + P.to_markdown(index=False) + '\n\n'
                + M.to_markdown(index=False) + '\n\n' + MP.to_markdown(index=False) + '\n')
    # budget curves (episodes per encounter) and equal-budget view (total transitions per seed, incl. PPO training
    # and checkpoint selection; search spend up to b approximated by attempts x mean transitions per attempt).
    # One style per method in every panel, one legend, panels from the sanity case to the strongest SUT.
    order = [x for x in SUT_ORDER if x in set(df['sut'])] + sorted(set(df['sut']) - set(SUT_ORDER))
    fig, axs = plt.subplots(2, len(order), figsize=(4.2 * len(order), 7.4), sharey=True, squeeze=False)
    handles = {}
    for j, sut in enumerate(order):
        for m, g in df[df['sut'] == sut].groupby('method'):
            st = method_style(m)
            B = int(g['budget'].iloc[0])
            bs_ = np.unique(np.round(np.logspace(0, np.log10(B), 40)).astype(int))
            curve = [((g['found'] == 1) & (g['first_attempt'] <= b)).groupby(g['seed']).mean().mean() for b in bs_]
            per_try = g['transitions'] / g['attempts'].clip(lower=1)
            fixed = g.groupby('seed')[['train_transitions', 'select_transitions']].first().sum(axis=1).mean()
            spend = [(np.minimum(g['attempts'], b) * per_try).groupby(g['seed']).sum().mean() + fixed for b in bs_]
            for ax, x in ((axs[0][j], bs_), (axs[1][j], spend)):
                h, = ax.plot(x, curve, **st)
            handles.setdefault(st['label'], h)
        axs[0][j].set_title('SUT: %s' % SUT_TITLE.get(sut, sut), fontsize=10)
        axs[0][j].set_xlabel('search budget, episodes per encounter')
        axs[1][j].set_xlabel('total transitions per seed\n(search + PPO training and checkpoint selection)')
        for ax in axs[:, j]:
            ax.set_xscale('log'); ax.grid(alpha=.3)
    for ax in axs[:, 0]:
        ax.set_ylabel('failure discovery rate\n(held-out encounters with an induced failure)')
    labels = [l for l in LEGEND_ORDER if l in handles]
    fig.legend([handles[l] for l in labels], labels, loc='lower center', ncol=min(len(labels), 9), fontsize=8,
               frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(os.path.join(args.out, 'budget_curves.png'), dpi=130)
    pd.set_option('display.width', 250)
    print(S.to_string(index=False))
    print(P.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run', 'report', 'envelope'])
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
    ap.add_argument('--ppo_root', default='', help='report: PPO run folders, to recompute the full training cost')
    ap.add_argument('--extra', default='', help='report: comma list of further compare_b<budget> folders')
    ap.add_argument('--v_list', default='6,7.5,9,12', help='envelope: attacker top speeds [m/s]')
    ap.add_argument('--suts', default='fixed,manual,autonomous,replan', help='envelope: systems under test')
    args = ap.parse_args()
    {'run': run, 'report': report, 'envelope': envelope}[args.cmd](args)


if __name__ == '__main__':
    main()
