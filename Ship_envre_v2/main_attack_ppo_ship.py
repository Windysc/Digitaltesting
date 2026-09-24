"""
Regenerated main script: attack-scenario PPO on the ShipAI-derived vessel,
structured after Desktop/main_attack_ppo.py (episode loop, periodic evaluation
with success rate, best checkpoint by success rate, txt/csv logs, scene
figures) but driving ShipAttackEnv from ship_env_v2.py instead of the
MassTestingEnv, and using the generated AIS trace sets as scenario input.
Success = a dangerous encounter by maritime criteria (encounter.py), not a
capture radius.  main_ppo_ship.py is the navigation counterpart
(Desktop/main_ppo.py) on the same environment with --task navigate.

Usage
-----
  # train on synthetic traces (created automatically if the data paths are missing)
  python main_attack_ppo_ship.py --save_dir runs/attack_v2 --num_episodes 3000

  # train on the generated data of the original pipeline
  python main_attack_ppo_ship.py --guideline path/dataset_1.csv.npy --mergeline path/dataset_2.csv.npy

  # evaluate a checkpoint only
  python main_attack_ppo_ship.py --mode eval --ckpt runs/attack_v2/models/PPO_ShipAttackEnv_seed0_best.pth
"""
import os
import csv
import json
import copy
import argparse
from datetime import datetime

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except Exception:                       # tqdm optional
    def tqdm(x, **k):
        return x

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, HERE)

import ppo_agent
from ppo_agent import PPO
from ship_env_v2 import ShipAttackEnv, OwnShipInit, TargetShipSpec, pursuit_policy, guideline_policy, OBS_NAMES
from encounter import EncounterParams
from target_ship import AUTOMATION_LEVELS
from grading import summarise
from make_synthetic_data import ensure_data


# ------------------------------------------------------------------ helpers
def plot_returns(returns, path, title='Training...', window=20):
    plt.figure(figsize=(8, 4))
    plt.title(title); plt.xlabel('Episode'); plt.ylabel('Episode return')
    r = np.asarray(returns, dtype=float)
    plt.plot(r, alpha=0.4, label='return')
    if len(r) >= window:
        ma = np.convolve(r, np.ones(window) / window, mode='valid')
        plt.plot(np.arange(window - 1, len(r)), ma, label='%d-episode mean' % window)
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=110); plt.close()


def load_policy(ckpt, task, continuous, args):
    """A trained checkpoint as a callable obs -> action (used for the other ship)."""
    state_dim = len(OBS_NAMES)
    action_dim = 2 if continuous else 15
    agent = PPO(state_dim, action_dim, args.lr_actor, args.lr_critic, args.gamma, args.K_epochs, args.eps_clip,
                bool(continuous), 0.6)
    agent.load(ckpt)
    print('other ship: policy %s (%s task)' % (ckpt, task))
    return lambda obs: agent.select_action(obs, deterministic=True)


def build_env(args, seed, evaluation=False):
    own = OwnShipInit(speed_range=(args.own_speed_min, args.own_speed_max),
                      heading_noise_deg=args.heading_noise_deg)
    overrides = {}
    if args.target_evasion is not None:
        overrides['evasion'] = args.target_evasion
    if args.evade_angle is not None:
        overrides['evade_angle_deg'] = args.evade_angle
    if args.alert_dcpa is not None:
        overrides['alert_dcpa'] = args.alert_dcpa
    if args.alert_tcpa is not None:
        overrides['alert_tcpa'] = args.alert_tcpa
    if args.latency is not None:
        overrides['latency'] = args.latency
    automation = args.automation if args.automation != 'agent' else 'assisted'
    tgt = TargetShipSpec(reverse=not args.target_forward, trace_dt=args.trace_dt,
                         speed_scale_range=(args.target_speed_min, args.target_speed_max),
                         smooth_sigma=args.smooth_sigma, yaw_rate_max_deg=args.target_yaw_rate,
                         lookahead=args.target_lookahead, automation=automation, overrides=overrides,
                         clear_steps=args.clear_steps)
    enc = EncounterParams(d_safe=args.d_safe, t_safe=args.t_safe, domain_a_L=args.domain_a,
                          domain_b_L=args.domain_b, collision_L=args.collision_L)
    other_policy = None
    if args.other_policy:
        other_policy = load_policy(args.other_policy, args.other_task, args.continuous, args)
    scenario = [x.strip() for x in args.scenario.split(',')] if ',' in args.scenario else args.scenario
    return ShipAttackEnv(args.guideline, args.mergeline, task=args.task, own_init=own, target_spec=tgt,
                         encounter=enc, scenario=scenario, cycle_scenarios=evaluation, other_policy=other_policy,
                         decision_interval=args.decision_interval, max_steps=args.max_ep_len,
                         border_margin=args.border_margin,
                         action_mode='continuous' if args.continuous else 'discrete',
                         reward_type=args.reward_type, success_severity=args.success_severity,
                         hold_steps=args.hold_steps, dest_radius=args.dest_radius,
                         dest_heading_tol_deg=args.dest_heading_tol, rudder_change_cost=args.rudder_change_cost,
                         save_dir=os.path.join(args.save_dir, 'scenes'), seed=seed)


def baseline_policy(env, obs):
    return pursuit_policy(env, obs) if env.task == 'attack' else guideline_policy(env, obs)


def run_episode(env, agent, max_ep_len, deterministic=True, store=False):
    """Roll out one episode. Returns (return, steps, success, eval_dict)."""
    state = env.reset()
    eps_return, eps_success = 0.0, 0
    for _ in range(max_ep_len):
        if agent is None:
            action = baseline_policy(env, state)
        else:
            action = agent.select_action(state, deterministic=deterministic)
        state, reward, done, success_flag = env.step(action)
        eps_return += reward
        if store:
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)
        if done:
            eps_success = int(success_flag == 1)
            break
    return eps_return, env.destination_step, eps_success, env.evaluation()


def evaluate(env, agent, args, tag, eval_log_dir, scene_budget=10):
    returns, steps, succ, dicts = [], [], [], []
    scenes = 0
    for i_eval in range(args.num_eval):
        r, s, ok, d = run_episode(env, agent, args.max_ep_len, deterministic=True)
        returns.append(r); steps.append(s); succ.append(ok); d['eval_index'] = i_eval; dicts.append(d)
        if ok and scenes < scene_budget:
            env.show_scenes(os.path.join(eval_log_dir, 'scene_%s_eval%d.png' % (tag, i_eval)))
            scenes += 1
    if scenes == 0 and len(dicts):          # always keep one failure scene for diagnosis
        env.show_scenes(os.path.join(eval_log_dir, 'scene_%s_last_fail.png' % tag))
    with open(os.path.join(eval_log_dir, 'eval_dicts_%s.csv' % tag), 'w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=list(dicts[0].keys()))
        w.writeheader(); w.writerows(dicts)
    # per-scenario breakdown with event grades
    lines = ['%-20s %5s %8s %10s %8s  grades(0/1/2/3)' % ('scenario', 'n', 'success', 'mean_score', 'steps')]
    for sc in sorted(set(d['scenario'] for d in dicts)):
        sub = [d for d in dicts if d['scenario'] == sc]
        sm = summarise(sub, env.task)
        gh = sm['grade_hist']
        lines.append('%-20s %5d %8.2f %10.1f %8.1f  %d/%d/%d/%d' % (
            sc, sm['n'], sm['success_rate'], sm['mean_score'], sm['mean_steps'],
            gh['safe_passage'], gh['close_quarters'], gh['domain_infringement'], gh['collision']))
    sm = summarise(dicts, env.task)
    lines.append('%-20s %5d %8.2f %10.1f %8.1f' % ('ALL', sm['n'], sm['success_rate'], sm['mean_score'], sm['mean_steps']))
    with open(os.path.join(eval_log_dir, 'eval_scenarios_%s.txt' % tag), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join('   ' + l for l in lines))
    return float(np.mean(returns)), float(np.mean(steps)), float(np.mean(succ)), dicts


# ---------------------------------------------------------------- training
def train(args):
    print('=' * 92)
    env_name = 'ShipAttackEnv' if args.task == 'attack' else 'ShipNavigateEnv'
    has_continuous_action_space = bool(args.continuous)
    max_ep_len = args.max_ep_len
    update_timestep = max_ep_len * args.update_every_episodes
    K_epochs, eps_clip, gamma = args.K_epochs, args.eps_clip, args.gamma
    lr_actor, lr_critic = args.lr_actor, args.lr_critic
    action_std, action_std_decay_rate, min_action_std = 0.6, 0.05, 0.1
    action_std_decay_freq = int(2.5e5)
    random_seed = args.seed

    save_dir = args.save_dir
    model_dir = os.path.join(save_dir, 'models')
    train_log_dir = os.path.join(save_dir, 'train_logs')
    eval_log_dir = os.path.join(save_dir, 'eval_logs')
    for d in (save_dir, model_dir, train_log_dir, eval_log_dir):
        os.makedirs(d, exist_ok=True)
    with open(os.path.join(save_dir, 'args.json'), 'wt') as f:
        json.dump(vars(args), f, indent=4)

    # seed unconditionally (the reference skips seeding when seed == 0, which
    # made 300-episode runs land anywhere between 0.15 and 0.95 success)
    import random as _random
    torch.manual_seed(random_seed); np.random.seed(random_seed); _random.seed(random_seed)
    set_device(args)
    env = build_env(args, seed=random_seed)
    eval_env = build_env(args, seed=random_seed + 12345, evaluation=True)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] if has_continuous_action_space else env.action_space.n
    checkpoint_path = os.path.join(model_dir, 'PPO_%s_seed%d.pth' % (env_name, random_seed))

    print('training environment name :', env_name, '| task', args.task)
    print('danger criteria:', env.enc_params.as_dict())
    print('scenarios: %s | other ship automation: %s%s' % (
        env.scenarios, args.automation, ' (policy %s)' % args.other_policy if args.other_policy else ''))
    print('traces: %d guidelines x %d mergelines, map %.0f x %.0f m' % (env.n_guide, env.n_merge, *env.map_size))
    print('state dim %d | action dim %d (%s) | reward %s' % (
        state_dim, action_dim, 'continuous' if has_continuous_action_space else 'discrete', args.reward_type))
    print('PPO update every %d steps | K %d | clip %.2f | gamma %.3f | lr %.1e / %.1e' % (
        update_timestep, K_epochs, eps_clip, gamma, lr_actor, lr_critic))
    print('checkpoint :', checkpoint_path)
    print('=' * 92)

    with open(os.path.join(train_log_dir, 'train_result.txt'), 'w') as f:
        f.write('Episode Return Steps Success Termination MinDist\n')
    with open(os.path.join(eval_log_dir, 'eval_result.txt'), 'w') as f:
        f.write('Episode AverageReturn Lens SuccessRate\n')

    # scripted baseline so the learned success rate has a reference point
    _, _, base_succ, _ = evaluate(eval_env, None, args, 'baseline', eval_log_dir, scene_budget=2)
    print('scripted baseline (%s) success rate on eval env: %.2f' % (
        'pure pursuit' if args.task == 'attack' else 'guideline following + starboard evasion', base_succ))

    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                    has_continuous_action_space, action_std, entropy_coef=args.entropy_coef,
                    minibatch_size=args.minibatch)
    if args.ckpt:
        ppo_agent.load(args.ckpt); print('resumed from', args.ckpt)

    start_time = datetime.now().replace(microsecond=0)
    print('Started training at :', start_time)

    time_step, best_eval_success = 0, -1.0
    train_returns, print_running_reward, print_running_episodes = [], 0.0, 0
    recent_success = []

    for i_episode in tqdm(range(1, args.num_episodes + 1)):
        state = env.reset()
        eps_return, success_flag = 0.0, 0
        for t in range(1, max_ep_len + 1):
            action = ppo_agent.select_action(state)
            state, reward, done, success_flag = env.step(action)
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
            time_step += 1
            eps_return += reward
            if time_step % update_timestep == 0:
                ppo_agent.update()
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)
            if done:
                break

        success = int(success_flag == 1)
        recent_success.append(success)
        train_returns.append(eps_return)
        with open(os.path.join(train_log_dir, 'train_result.txt'), 'a') as f:
            f.write('%d %.3f %d %d %s %.0f\n' % (i_episode, eps_return, env.destination_step, success,
                                                env.termination, env.min_distance))
        print_running_reward += eps_return; print_running_episodes += 1

        if i_episode % args.print_every == 0:
            print('Episode %d | timestep %d | avg return %.2f | success rate (last %d) %.2f' % (
                i_episode, time_step, print_running_reward / print_running_episodes,
                len(recent_success[-args.print_every:]), np.mean(recent_success[-args.print_every:])))
            print_running_reward, print_running_episodes = 0.0, 0
            plot_returns(train_returns, os.path.join(save_dir, 'train_rl_returns.png'))

        if i_episode % args.eval_every == 0:
            ppo_agent.save(checkpoint_path)
            ppo_agent.save(checkpoint_path.replace('.pth', '_episode%d.pth' % i_episode))
            test_agent = copy.deepcopy(ppo_agent)
            mr, ms, sr, _ = evaluate(eval_env, test_agent, args, 'episode%d' % i_episode, eval_log_dir)
            with open(os.path.join(eval_log_dir, 'eval_result.txt'), 'a') as f:
                f.write('%d %.3f %.1f %.3f\n' % (i_episode, mr, ms, sr))
            print('Eval @ episode %d: return %.2f, steps %.1f, success rate %.2f (baseline %.2f)' % (
                i_episode, mr, ms, sr, base_succ))
            if sr > best_eval_success:
                best_eval_success = sr
                ppo_agent.save(checkpoint_path.replace('.pth', '_best.pth'))
                print('  saved best checkpoint (success rate %.2f)' % sr)
            print('  elapsed', datetime.now().replace(microsecond=0) - start_time)

    ppo_agent.save(checkpoint_path)
    plot_returns(train_returns, os.path.join(save_dir, 'train_rl_returns.png'), title='Training returns')
    print('=' * 92)
    print('Finished at :', datetime.now().replace(microsecond=0), '| total', datetime.now().replace(microsecond=0) - start_time)
    print('best eval success rate %.2f (scripted baseline %.2f)' % (best_eval_success, base_succ))
    print('=' * 92)


def set_device(args):
    """--device auto|cpu|cuda ; the agent module reads its global `device` at call time."""
    if args.device == 'auto':
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        dev = args.device
    if dev.startswith('cuda') and not torch.cuda.is_available():
        print('WARNING: --device cuda requested but torch.cuda.is_available() is False; using cpu')
        dev = 'cpu'
    ppo_agent.device = torch.device(dev)
    print('torch %s | device %s%s' % (torch.__version__, dev,
          ' (%s)' % torch.cuda.get_device_name(0) if dev.startswith('cuda') else ''))


def eval_only(args):
    set_device(args)
    os.makedirs(args.save_dir, exist_ok=True)
    eval_log_dir = os.path.join(args.save_dir, 'eval_only_logs'); os.makedirs(eval_log_dir, exist_ok=True)
    env = build_env(args, seed=args.seed + 999, evaluation=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] if args.continuous else env.action_space.n
    agent = None
    if args.ckpt:
        agent = PPO(state_dim, action_dim, args.lr_actor, args.lr_critic, args.gamma, args.K_epochs,
                    args.eps_clip, bool(args.continuous), 0.6)
        agent.load(args.ckpt)
    tag = 'ckpt' if agent is not None else 'baseline'
    mr, ms, sr, dicts = evaluate(env, agent, args, tag, eval_log_dir, scene_budget=args.num_eval)
    with open(os.path.join(eval_log_dir, 'summary_%s.json' % tag), 'w') as f:
        json.dump(dict(mean_return=mr, mean_steps=ms, success_rate=sr,
                       terminations={k: sum(d['termination'] == k for d in dicts) for k in set(d['termination'] for d in dicts)}),
                  f, indent=2)
    print('%s: return %.2f | steps %.1f | success rate %.2f over %d episodes' % (tag, mr, ms, sr, args.num_eval))


# ---------------------------------------------------------------- arguments
def gen_args(argv=None, task_default='attack'):
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['train', 'eval'], default='train')
    p.add_argument('--task', choices=['attack', 'navigate'], default=task_default)
    p.add_argument('--save_dir', type=str, default=os.path.join(HERE, 'runs', '%s_v2' % task_default))
    p.add_argument('--ckpt', type=str, default='', help='checkpoint to evaluate or resume from')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', type=str, default='auto', help='auto | cpu | cuda')
    # data
    p.add_argument('--guideline', type=str, default='', help='(n,100,2) lon/lat .npy for the own-ship reference')
    p.add_argument('--mergeline', type=str, default='', help='(n,100,2) lon/lat .npy for the target trace')
    p.add_argument('--data_dir', type=str, default=os.path.join(HERE, 'data'))
    p.add_argument('--trace_dt', type=float, default=20.0, help='seconds between trace points')
    p.add_argument('--target_forward', action='store_true', help='walk the merge line forwards instead of reversed')
    p.add_argument('--target_speed_min', type=float, default=0.8)
    p.add_argument('--target_speed_max', type=float, default=1.2)
    p.add_argument('--own_speed_min', type=float, default=3.0)
    p.add_argument('--own_speed_max', type=float, default=10.0)
    p.add_argument('--heading_noise_deg', type=float, default=10.0)
    # scenarios (scenarios.py) and the other ship (target_ship.py)
    p.add_argument('--scenario', type=str, default='data',
                   help="'data', 'mix', one of head_on/crossing_starboard/crossing_port/overtaking/overtaken, or a comma list")
    p.add_argument('--automation', type=str, default='assisted', choices=list(AUTOMATION_LEVELS) + ['agent'],
                   help='other-ship automation level; agent = policy from --other_policy')
    p.add_argument('--other_policy', type=str, default='', help='checkpoint that drives the other ship (self-play)')
    p.add_argument('--other_task', type=str, default='', help='task of that policy (default: the opposite task)')
    p.add_argument('--smooth_sigma', type=float, default=2.0, help='Gaussian smoothing of traces into routes [points]')
    p.add_argument('--target_yaw_rate', type=float, default=0.5, help='target max turn rate [deg/s]')
    p.add_argument('--target_lookahead', type=float, default=600.0, help='pure-pursuit look-ahead [m]')
    p.add_argument('--target_evasion', type=str, default=None, choices=[None, 'colreg', 'none'],
                   help='override the level: colreg | none')
    p.add_argument('--evade_angle', type=float, default=None, help='override: starboard alteration [deg]')
    p.add_argument('--alert_dcpa', type=float, default=None, help='override: evasion trigger DCPA [m]')
    p.add_argument('--alert_tcpa', type=float, default=None, help='override: evasion trigger TCPA [s]')
    p.add_argument('--latency', type=float, default=None, help='override: reaction time [s]')
    p.add_argument('--clear_steps', type=int, default=6, help='clear checks before the target resumes its route')
    # danger criteria (encounter.py)
    p.add_argument('--d_safe', type=float, default=926.0, help='DCPA threshold of the CPA warning [m]')
    p.add_argument('--t_safe', type=float, default=900.0, help='TCPA window of the CPA warning [s]')
    p.add_argument('--domain_a', type=float, default=4.0, help='ship-domain semi-axis along course [ship lengths]')
    p.add_argument('--domain_b', type=float, default=1.6, help='ship-domain semi-axis across course [ship lengths]')
    p.add_argument('--collision_L', type=float, default=1.0, help='collision distance [ship lengths]')
    p.add_argument('--success_severity', type=int, default=2, help='attack: severity level that counts as success')
    p.add_argument('--hold_steps', type=int, default=2, help='attack: steps the severity must be held')
    p.add_argument('--dest_radius', type=float, default=200.0, help='navigate: destination radius [m]')
    p.add_argument('--dest_heading_tol', type=float, default=15.0, help='navigate: heading tolerance at destination [deg]')
    # environment
    p.add_argument('--decision_interval', type=float, default=10.0)
    p.add_argument('--max_ep_len', type=int, default=300)
    p.add_argument('--border_margin', type=float, default=1500.0)
    p.add_argument('--reward_type', type=str, default='shaped', choices=['shaped', 'final_attack_reward'])
    p.add_argument('--rudder_change_cost', type=float, default=0.05, help='reward cost per unit rudder-level change')
    p.add_argument('--continuous', type=int, default=0, help='1 = continuous rudder/throttle, 0 = 15 discrete actions')
    # PPO
    p.add_argument('--num_episodes', type=int, default=3000)
    p.add_argument('--update_every_episodes', type=int, default=4, help='PPO update every max_ep_len * this steps')
    p.add_argument('--K_epochs', type=int, default=20)
    p.add_argument('--minibatch', type=int, default=256)
    p.add_argument('--eps_clip', type=float, default=0.2)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--lr_actor', type=float, default=3e-4)
    p.add_argument('--lr_critic', type=float, default=1e-3)
    p.add_argument('--entropy_coef', type=float, default=0.01)
    # logging / evaluation
    p.add_argument('--print_every', type=int, default=20)
    p.add_argument('--eval_every', type=int, default=100)
    p.add_argument('--num_eval', type=int, default=50)
    args = p.parse_args(argv)

    if not args.other_task:
        args.other_task = 'navigate' if args.task == 'attack' else 'attack'
    if args.automation == 'agent' and not args.other_policy:
        p.error('--automation agent needs --other_policy <checkpoint>')
    if not args.guideline or not args.mergeline:
        gp, mp = ensure_data(args.data_dir, seed=args.seed)
        args.guideline = args.guideline or gp
        args.mergeline = args.mergeline or mp
        print('using trace sets:', args.guideline, args.mergeline)
    return args


def run(task_default='attack'):
    args = gen_args(task_default=task_default)
    if args.mode == 'train':
        train(args)
    else:
        eval_only(args)


if __name__ == '__main__':
    run('attack')
