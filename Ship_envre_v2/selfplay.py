"""
Self-play between the attack agent and the navigation agent.

Round r:
  1. train the attack agent against the other ship driven by the latest
     navigation policy (round 1: the rule-based 'assisted' target);
  2. train the navigation agent against the other ship driven by the latest
     attack policy.
Each round warm-starts from the previous checkpoint of the same task.  All
arguments after the script's own are passed to main_attack_ppo_ship.gen_args
(episodes per round, scenarios, danger criteria, data files, ...).

  python selfplay.py --rounds 3 --out runs/selfplay --num_episodes 600 --scenario mix
"""
import os
import sys
import shutil
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import main_attack_ppo_ship as M


def best_ckpt(save_dir, env_name, seed):
    p = os.path.join(save_dir, 'models', 'PPO_%s_seed%d_best.pth' % (env_name, seed))
    if not os.path.exists(p):
        p = os.path.join(save_dir, 'models', 'PPO_%s_seed%d.pth' % (env_name, seed))
    return p


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--rounds', type=int, default=2)
    p.add_argument('--out', type=str, default=os.path.join(HERE, 'runs', 'selfplay'))
    p.add_argument('--start_attack', type=str, default='', help='attack checkpoint to start from')
    p.add_argument('--start_navigate', type=str, default='', help='navigation checkpoint to start from')
    a, rest = p.parse_known_args()
    os.makedirs(a.out, exist_ok=True)
    att_ckpt, nav_ckpt = a.start_attack, a.start_navigate

    for r in range(1, a.rounds + 1):
        # ---- attack vs navigation policy -------------------------------------
        argv = list(rest) + ['--task', 'attack', '--save_dir', os.path.join(a.out, 'round%d_attack' % r)]
        if nav_ckpt:
            argv += ['--automation', 'agent', '--other_policy', nav_ckpt, '--other_task', 'navigate']
        if att_ckpt:
            argv += ['--ckpt', att_ckpt]
        args = M.gen_args(argv, task_default='attack')
        print('=' * 30, 'round %d: attack agent vs %s' % (r, nav_ckpt or 'rule-based target'), '=' * 30)
        M.train(args)
        att_ckpt = best_ckpt(args.save_dir, 'ShipAttackEnv', args.seed)

        # ---- navigation vs attack policy -------------------------------------
        argv = list(rest) + ['--task', 'navigate', '--save_dir', os.path.join(a.out, 'round%d_navigate' % r),
                             '--automation', 'agent', '--other_policy', att_ckpt, '--other_task', 'attack']
        if nav_ckpt:
            argv += ['--ckpt', nav_ckpt]
        args = M.gen_args(argv, task_default='navigate')
        print('=' * 30, 'round %d: navigation agent vs %s' % (r, att_ckpt), '=' * 30)
        M.train(args)
        nav_ckpt = best_ckpt(args.save_dir, 'ShipNavigateEnv', args.seed)

    shutil.copy(att_ckpt, os.path.join(a.out, 'attack_final.pth'))
    shutil.copy(nav_ckpt, os.path.join(a.out, 'navigate_final.pth'))
    print('self-play done:', att_ckpt, nav_ckpt)


if __name__ == '__main__':
    main()
