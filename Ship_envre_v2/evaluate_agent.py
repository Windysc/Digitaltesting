"""
Score an agent (or the scripted baseline) over a matrix of COLREG scenarios
x other-ship automation levels, marking every episode with the event grade
and event score of grading.py.

  python evaluate_agent.py --task attack --ckpt checkpoints/ppo_attack_best.pth \\
         --scenarios mix --automation none,manual,assisted,autonomous --n 10 --out scorecards/attack
  python evaluate_agent.py --task navigate --n 10 --out scorecards/navigate_baseline     # baseline
  python evaluate_agent.py --task attack --ckpt <ckpt> --automation agent --other_policy checkpoints/ppo_navigate_best.pth

Outputs in --out:
  episodes.csv        one row per episode (all evaluation() fields + grade + score)
  scorecard.md        table scenario x automation: success rate / mean event score / grade histogram
  scorecard.json      the same numbers plus the ability index
  heatmap_success.png, heatmap_score.png

Ability index (attack): mean success rate over the matrix, mean event score,
efficiency vs the scripted baseline (steps), manoeuvre effort, robustness =
min success rate over automation levels.  For the navigation task the
safety score (100 - event score) and the COLREG-compliance rate replace the
event score.

Extra arguments are passed to main_attack_ppo_ship.gen_args (environment
configuration), e.g. --d_safe 1200 --guideline <..>.npy --mergeline <..>.npy
"""
import os
import sys
import csv
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import main_attack_ppo_ship as M
from scenarios import SCENARIOS
from grading import summarise, GRADE_NAMES


def run_cell(args, agent, scenario, automation, n, seed):
    args.scenario = scenario
    args.automation = automation
    env = M.build_env(args, seed=seed, evaluation=True)
    recs = []
    for _ in range(n):
        _, _, _, d = M.run_episode(env, agent, args.max_ep_len, deterministic=True)
        recs.append(d)
    return recs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', type=str, default='', help='checkpoint; omit for the scripted baseline')
    p.add_argument('--scenarios', type=str, default='mix', help="'mix' or comma list of scenario names")
    p.add_argument('--automation_levels', type=str, default='none,manual,assisted,autonomous',
                   help="comma list; add 'agent' with --other_policy")
    p.add_argument('--n', type=int, default=10, help='episodes per scenario per automation level')
    p.add_argument('--out', type=str, default=os.path.join(HERE, 'scorecards', 'run'))
    p.add_argument('--eval_seed', type=int, default=2024)
    p.add_argument('--no_baseline', action='store_true', help='skip the scripted-baseline reference run')
    a, rest = p.parse_known_args()
    args = M.gen_args(rest)
    args.device = 'cpu'
    M.set_device(args)
    os.makedirs(a.out, exist_ok=True)

    scen = SCENARIOS if a.scenarios == 'mix' else [x.strip() for x in a.scenarios.split(',')]
    levels = [x.strip() for x in a.automation_levels.split(',')]
    agent = None
    if a.ckpt:
        state_dim = len(M.OBS_NAMES)
        action_dim = 2 if args.continuous else 15
        agent = M.PPO(state_dim, action_dim, args.lr_actor, args.lr_critic, args.gamma, args.K_epochs, args.eps_clip,
                      bool(args.continuous), 0.6)
        agent.load(a.ckpt)
    label = os.path.basename(a.ckpt) if a.ckpt else 'scripted baseline'
    print('scoring %s | task %s | scenarios %s | automation %s | %d episodes per cell' % (
        label, args.task, scen, levels, a.n))

    rows, cells = [], {}
    base_steps = {}
    for lv in levels:
        for sc in scen:
            recs = run_cell(args, agent, sc, lv, a.n, a.eval_seed)
            for r in recs:
                r['automation_level'] = lv
            rows.extend(recs)
            if agent is not None and not a.no_baseline:
                b = run_cell(args, None, sc, lv, a.n, a.eval_seed)
                ok = [x['destination_step'] for x in b if x['success']]
                base_steps[(sc, lv)] = float(np.mean(ok)) if ok else None
            cells[(sc, lv)] = summarise(recs, args.task, base_steps.get((sc, lv)))
            c = cells[(sc, lv)]
            print('  %-20s %-11s success %.2f  score %5.1f  grades %s%s' % (
                sc, lv, c['success_rate'], c['mean_score'], c['grade_hist'],
                '  eff %.2f' % c['efficiency'] if 'efficiency' in c else ''))

    # ---- files ------------------------------------------------------------
    with open(os.path.join(a.out, 'episodes.csv'), 'w', newline='', encoding='utf8') as f:
        keys = sorted(set().union(*[r.keys() for r in rows]))
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    succ = np.array([[cells[(sc, lv)]['success_rate'] for lv in levels] for sc in scen])
    score = np.array([[cells[(sc, lv)]['mean_score'] for lv in levels] for sc in scen])
    overall = summarise(rows, args.task)
    ability = dict(label=label, task=args.task, n_episodes=len(rows),
                   success_rate=float(succ.mean()), mean_event_score=float(score.mean()),
                   robustness_min_success_over_automation=float(succ.mean(axis=0).min()),
                   hardest_scenario=scen[int(succ.mean(axis=1).argmin())],
                   hardest_automation=levels[int(succ.mean(axis=0).argmin())],
                   effort_rudder_change_per_step=overall['effort'],
                   grade_hist=overall['grade_hist'])
    effs = [c['efficiency'] for c in cells.values() if 'efficiency' in c]
    if effs:
        ability['efficiency_vs_baseline'] = float(np.mean(effs))
    if args.task == 'navigate':
        ability['safety_score'] = overall.get('safety_score')
        ability['colreg_compliance'] = overall.get('colreg_compliance')

    md = ['# Scorecard: %s (%s task)' % (label, args.task), '',
          '%d episodes per cell, seed %d.  Cell = success rate / mean event score / grades 0-1-2-3.' % (a.n, a.eval_seed), '',
          '| scenario | ' + ' | '.join(levels) + ' |', '|---|' + '---|' * len(levels)]
    for sc in scen:
        parts = []
        for lv in levels:
            c = cells[(sc, lv)]
            g = c['grade_hist']
            parts.append('%.2f / %.0f / %d-%d-%d-%d' % (c['success_rate'], c['mean_score'], g['safe_passage'],
                                                       g['close_quarters'], g['domain_infringement'], g['collision']))
        md.append('| %s | ' % sc + ' | '.join(parts) + ' |')
    md += ['', '## Ability index', '']
    for k, v in ability.items():
        md.append('- %s: %s' % (k, v))
    with open(os.path.join(a.out, 'scorecard.md'), 'w', encoding='utf8') as f:
        f.write('\n'.join(md) + '\n')
    with open(os.path.join(a.out, 'scorecard.json'), 'w') as f:
        json.dump(dict(ability=ability, cells={'%s|%s' % k: v for k, v in cells.items()}), f, indent=2, default=float)

    for name, mat, cmap, vmin, vmax in [('success', succ, 'Blues', 0, 1), ('score', score, 'Reds', 0, 100)]:
        fig, ax = plt.subplots(figsize=(1.6 * len(levels) + 2.5, 0.5 * len(scen) + 1.5))
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xticks(range(len(levels))); ax.set_xticklabels(levels)
        ax.set_yticks(range(len(scen))); ax.set_yticklabels(scen)
        for i in range(len(scen)):
            for j in range(len(levels)):
                ax.text(j, i, '%.2f' % mat[i, j] if name == 'success' else '%.0f' % mat[i, j],
                        ha='center', va='center', fontsize=8, color='k')
        ax.set_title('%s: %s (%s)' % (label, 'success rate' if name == 'success' else 'mean event score', args.task))
        ax.set_xlabel('other-ship automation level')
        fig.colorbar(im, ax=ax, fraction=0.04)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, 'heatmap_%s.png' % name), dpi=120); plt.close(fig)
    print('\n' + '\n'.join(md[-len(ability) - 2:]))
    print('written to', a.out)


if __name__ == '__main__':
    main()
