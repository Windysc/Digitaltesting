"""
Training / evaluation runner for ShipEncounterEnv (v3, encounter-lifecycle
definition).  Same structure as Desktop/main_attack_ppo.py and
Ship_envre_v2/main_attack_ppo_ship.py (episode loop, periodic evaluation,
best checkpoint by success rate, txt/csv logs, scene figures), but:

  * no destination point and no timeout failure: episodes end with the
    encounter lifecycle (see ship_env_v3.py); the safety cap ends an episode
    as 'unresolved' (flag 0), reported separately;
  * per-scenario evaluation reports resolution time and recovery time;
  * --render replays evaluation episodes in the live turtle viewer
    (viewer_v3.py) and --record_experiment saves a ShipExperiment pickle
    (ship_data_v3.py), the two SimpleShipAI elements.

Entry points: main_attack_ppo_v3.py (attack) and main_ppo_v3.py (navigate).
"""
import os
import sys
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
except Exception:
    def tqdm(x, **k):
        return x

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(os.path.dirname(HERE), 'Ship_envre_v2')
for p_ in (HERE, V2):
    if p_ not in sys.path:
        sys.path.insert(0, p_)

import ppo_agent                                                            # noqa: E402
from ppo_agent import PPO                                                   # noqa: E402
from ship_env_v3 import (ShipEncounterEnv, EncounterRules, OBS_NAMES, pursuit_policy,   # noqa: E402
                         guideline_policy, PHASE_NAMES)
from ship_env_v2 import OwnShipInit, TargetShipSpec                         # noqa: E402
from encounter import EncounterParams                                       # noqa: E402
from target_ship import AUTOMATION_LEVELS                                   # noqa: E402
from grading import summarise                                               # noqa: E402
from make_synthetic_data import ensure_data                                 # noqa: E402


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


def set_device(args):
    if args.device == 'auto':
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        dev = args.device
    if dev.startswith('cuda') and not torch.cuda.is_available():
        print('WARNING: --device cuda requested but torch.cuda.is_available() is False; using cpu')
        dev = 'cpu'
    ppo_agent.device = torch.device(dev)
    print('torch %s | device %s' % (torch.__version__, dev))


def load_policy(ckpt, task, continuous, args):
    agent = PPO(len(OBS_NAMES), 2 if continuous else 15, args.lr_actor, args.lr_critic, args.gamma,
                args.K_epochs, args.eps_clip, bool(continuous), 0.6)
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
    rules = EncounterRules(r_clear=args.r_clear, n_clear=args.n_clear, xte_tol=args.xte_tol,
                           course_tol_deg=args.course_tol, recover_steps=args.recover_steps,
                           abandon_xte=args.abandon_xte, abandon_steps=args.abandon_steps,
                           route_extension=args.route_extension)
    other_policy = load_policy(args.other_policy, args.other_task, args.continuous, args) if args.other_policy else None
    scenario = [x.strip() for x in args.scenario.split(',')] if ',' in args.scenario else args.scenario
    return ShipEncounterEnv(args.guideline, args.mergeline, task=args.task, own_init=own, target_spec=tgt,
                            encounter=enc, rules=rules, scenario=scenario, cycle_scenarios=evaluation,
                            other_policy=other_policy, route_shape=args.target_route_shape,
                            straighten_tol=args.straighten_tol, evasion_cooldown=args.evasion_cooldown,
                            decision_interval=args.decision_interval,
                            max_steps=args.max_ep_len, border_margin=args.border_margin,
                            action_mode='continuous' if args.continuous else 'discrete',
                            reward_type=args.reward_type, success_severity=args.success_severity,
                            hold_steps=args.hold_steps, rudder_change_cost=args.rudder_change_cost,
                            save_dir=os.path.join(args.save_dir, 'scenes'), seed=seed)


def baseline_policy(env, obs):
    return pursuit_policy(env, obs) if env.task == 'attack' else guideline_policy(env, obs)


def run_episode(env, agent, max_ep_len, deterministic=True, store=False, render=False):
    state = env.reset()
    eps_return, eps_success = 0.0, 0
    for _ in range(max_ep_len):
        action = baseline_policy(env, state) if agent is None else agent.select_action(state, deterministic=deterministic)
        state, reward, done, success_flag = env.step(action)
        eps_return += reward
        if render:
            env.render()
        if store:
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)
        if done:
            eps_success = int(success_flag == 1)
            break
    return eps_return, env.destination_step, eps_success, env.evaluation()


def _fmt_time(x):
    return '%6.0f' % x if x is not None and x >= 0 else '     -'


def evaluate(env, agent, args, tag, eval_log_dir, scene_budget=10, render=False):
    returns, steps, succ, dicts = [], [], [], []
    scenes = 0
    for i_eval in range(args.num_eval):
        r, s, ok, d = run_episode(env, agent, args.max_ep_len, deterministic=True, render=render)
        returns.append(r); steps.append(s); succ.append(ok); d['eval_index'] = i_eval; dicts.append(d)
        if ok and scenes < scene_budget:
            env.show_scenes(os.path.join(eval_log_dir, 'scene_%s_eval%d.png' % (tag, i_eval)))
            scenes += 1
    if scenes == 0 and len(dicts):
        env.show_scenes(os.path.join(eval_log_dir, 'scene_%s_last_fail.png' % tag))
    with open(os.path.join(eval_log_dir, 'eval_dicts_%s.csv' % tag), 'w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=list(dicts[0].keys()))
        w.writeheader(); w.writerows(dicts)
    lines = ['%-20s %4s %8s %6s %10s %9s %9s  grades(0/1/2/3)  terminations' % (
        'scenario', 'n', 'success', 'unres', 'mean_score', 'resol[s]', 'recov[s]')]
    for sc in sorted(set(d['scenario'] for d in dicts)) + ['ALL']:
        sub = [d for d in dicts if sc == 'ALL' or d['scenario'] == sc]
        sm = summarise(sub, env.task)
        gh = sm['grade_hist']
        res = [d['resolution_time'] for d in sub if d['resolution_time'] >= 0]
        rec = [d['recovery_time'] for d in sub if d['recovery_time'] >= 0]
        term = {}
        for d in sub:
            term[d['termination']] = term.get(d['termination'], 0) + 1
        lines.append('%-20s %4d %8.2f %6d %10.1f %9s %9s  %d/%d/%d/%d  %s' % (
            sc, sm['n'], sm['success_rate'], sum(d['unresolved'] for d in sub), sm['mean_score'],
            _fmt_time(np.mean(res) if res else -1), _fmt_time(np.mean(rec) if rec else -1),
            gh['safe_passage'], gh['close_quarters'], gh['domain_infringement'], gh['collision'],
            ' '.join('%s:%d' % kv for kv in sorted(term.items()))))
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
    update_timestep = args.update_every_steps
    K_epochs, eps_clip, gamma = args.K_epochs, args.eps_clip, args.gamma
    lr_actor, lr_critic = args.lr_actor, args.lr_critic
    action_std, action_std_decay_rate, min_action_std, action_std_decay_freq = 0.6, 0.05, 0.1, int(2.5e5)
    random_seed = args.seed

    save_dir = args.save_dir
    model_dir = os.path.join(save_dir, 'models')
    train_log_dir = os.path.join(save_dir, 'train_logs')
    eval_log_dir = os.path.join(save_dir, 'eval_logs')
    for d in (save_dir, model_dir, train_log_dir, eval_log_dir):
        os.makedirs(d, exist_ok=True)
    with open(os.path.join(save_dir, 'args.json'), 'wt') as f:
        json.dump(vars(args), f, indent=4)

    import random as _random
    torch.manual_seed(random_seed); np.random.seed(random_seed); _random.seed(random_seed)
    set_device(args)
    env = build_env(args, seed=random_seed)
    eval_env = build_env(args, seed=random_seed + 12345, evaluation=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] if has_continuous_action_space else env.action_space.n
    checkpoint_path = os.path.join(model_dir, 'PPO_%s_seed%d.pth' % (env_name, random_seed))

    print('environment: %s (v3 encounter lifecycle) | task %s' % (env_name, args.task))
    print('lifecycle rules:', env.rules.as_dict())
    print('danger criteria:', env.enc_params.as_dict())
    print('scenarios: %s | other ship automation: %s%s | target route %s, evasion cooldown %d steps' % (
        env.scenarios, args.automation, ' (policy %s)' % args.other_policy if args.other_policy else '',
        args.target_route_shape, args.evasion_cooldown))
    print('state dim %d | action dim %d | reward %s | PPO update every %d steps, K %d, clip %.2f, gamma %.3f' % (
        state_dim, action_dim, args.reward_type, update_timestep, K_epochs, eps_clip, gamma))
    print('checkpoint :', checkpoint_path)
    print('=' * 92)
    with open(os.path.join(train_log_dir, 'train_result.txt'), 'w') as f:
        f.write('Episode Return Steps Success Termination Phase MinDCPA Scenario\n')
    with open(os.path.join(eval_log_dir, 'eval_result.txt'), 'w') as f:
        f.write('Episode AverageReturn Lens SuccessRate\n')

    _, _, base_succ, _ = evaluate(eval_env, None, args, 'baseline', eval_log_dir, scene_budget=2)
    print('scripted baseline success rate on eval env: %.2f' % base_succ)

    ppo_agent_ = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                     has_continuous_action_space, action_std, entropy_coef=args.entropy_coef,
                     minibatch_size=args.minibatch)
    if args.ckpt:
        ppo_agent_.load(args.ckpt); print('resumed from', args.ckpt)

    start_time = datetime.now().replace(microsecond=0)
    time_step, best_eval_success = 0, -1.0
    train_returns, recent_success, print_reward, print_n = [], [], 0.0, 0
    for i_episode in tqdm(range(1, args.num_episodes + 1)):
        state = env.reset()
        eps_return, success_flag = 0.0, 0
        for t in range(1, max_ep_len + 1):
            action = ppo_agent_.select_action(state)
            state, reward, done, success_flag = env.step(action)
            ppo_agent_.buffer.rewards.append(reward)
            ppo_agent_.buffer.is_terminals.append(done)
            time_step += 1
            eps_return += reward
            if time_step % update_timestep == 0:
                ppo_agent_.update()
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent_.decay_action_std(action_std_decay_rate, min_action_std)
            if done:
                break
        success = int(success_flag == 1)
        recent_success.append(success); train_returns.append(eps_return)
        with open(os.path.join(train_log_dir, 'train_result.txt'), 'a') as f:
            f.write('%d %.3f %d %d %s %s %.0f %s\n' % (
                i_episode, eps_return, env.step_count, success, env.termination, PHASE_NAMES[env.phase],
                env.min_dcpa if np.isfinite(env.min_dcpa) else -1, env.scenario))
        print_reward += eps_return; print_n += 1
        if i_episode % args.print_every == 0:
            print('Episode %d | timestep %d | avg return %.2f | success rate (last %d) %.2f' % (
                i_episode, time_step, print_reward / print_n, len(recent_success[-args.print_every:]),
                np.mean(recent_success[-args.print_every:])))
            print_reward, print_n = 0.0, 0
            plot_returns(train_returns, os.path.join(save_dir, 'train_rl_returns.png'))
        if i_episode % args.eval_every == 0:
            ppo_agent_.save(checkpoint_path)
            ppo_agent_.save(checkpoint_path.replace('.pth', '_episode%d.pth' % i_episode))
            mr, ms, sr, _ = evaluate(eval_env, copy.deepcopy(ppo_agent_), args, 'episode%d' % i_episode, eval_log_dir)
            with open(os.path.join(eval_log_dir, 'eval_result.txt'), 'a') as f:
                f.write('%d %.3f %.1f %.3f\n' % (i_episode, mr, ms, sr))
            print('Eval @ episode %d: return %.2f, steps %.1f, success rate %.2f (baseline %.2f)' % (
                i_episode, mr, ms, sr, base_succ))
            if sr > best_eval_success:
                best_eval_success = sr
                ppo_agent_.save(checkpoint_path.replace('.pth', '_best.pth'))
                print('  saved best checkpoint (success rate %.2f)' % sr)
            print('  elapsed', datetime.now().replace(microsecond=0) - start_time)
    ppo_agent_.save(checkpoint_path)
    plot_returns(train_returns, os.path.join(save_dir, 'train_rl_returns.png'), title='Training returns')
    print('=' * 92)
    print('finished | total', datetime.now().replace(microsecond=0) - start_time,
          '| best eval success rate %.2f (baseline %.2f)' % (best_eval_success, base_succ))
    print('=' * 92)


def eval_only(args):
    set_device(args)
    os.makedirs(args.save_dir, exist_ok=True)
    eval_log_dir = os.path.join(args.save_dir, 'eval_only_logs'); os.makedirs(eval_log_dir, exist_ok=True)
    env = build_env(args, seed=args.seed + 999, evaluation=True)
    agent = None
    if args.ckpt:
        agent = PPO(env.observation_space.shape[0], env.action_space.shape[0] if args.continuous else env.action_space.n,
                    args.lr_actor, args.lr_critic, args.gamma, args.K_epochs, args.eps_clip, bool(args.continuous), 0.6)
        agent.load(args.ckpt)
    recorder = None
    if args.record_experiment:
        from ship_data_v3 import ShipExperiment
        recorder = ShipExperiment(info='%s eval %s' % (args.task, 'ckpt' if agent else 'baseline'),
                                  time_step=args.decision_interval)
        env.set_recorder(recorder)
    tag = 'ckpt' if agent is not None else 'baseline'
    mr, ms, sr, dicts = evaluate(env, agent, args, tag, eval_log_dir, scene_budget=args.num_eval, render=args.render)
    if recorder is not None:
        cwd = os.getcwd(); os.chdir(args.save_dir)
        name = recorder.save_experiment('%s_%s' % (args.task, tag))
        recorder.plot_trajectory(save_path=os.path.join('eval_only_logs', 'experiment_trajectories.png'))
        os.chdir(cwd)
    if args.render and env.viewer is not None:
        print('viewer window open; close it to finish')
        env.viewer.freeze_scream()
    with open(os.path.join(eval_log_dir, 'summary_%s.json' % tag), 'w') as f:
        json.dump(dict(mean_return=mr, mean_steps=ms, success_rate=sr,
                       unresolved=sum(d['unresolved'] for d in dicts),
                       terminations={k: sum(d['termination'] == k for d in dicts) for k in set(d['termination'] for d in dicts)}),
                  f, indent=2)
    print('%s: return %.2f | steps %.1f | success rate %.2f over %d episodes' % (tag, mr, ms, sr, args.num_eval))


# ---------------------------------------------------------------- arguments
def gen_args(argv=None, task_default='attack'):
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['train', 'eval'], default='train')
    p.add_argument('--task', choices=['attack', 'navigate'], default=task_default)
    p.add_argument('--save_dir', type=str, default=os.path.join(HERE, 'runs', '%s_v3' % task_default))
    p.add_argument('--ckpt', type=str, default='')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--render', action='store_true', help='eval: live turtle viewer (viewer_v3.py)')
    p.add_argument('--record_experiment', action='store_true', help='eval: save a ShipExperiment pickle (ship_data_v3.py)')
    # data
    p.add_argument('--guideline', type=str, default='')
    p.add_argument('--mergeline', type=str, default='')
    p.add_argument('--data_dir', type=str, default=os.path.join(V2, 'data'))
    p.add_argument('--trace_dt', type=float, default=20.0)
    p.add_argument('--target_forward', action='store_true')
    p.add_argument('--target_speed_min', type=float, default=0.8)
    p.add_argument('--target_speed_max', type=float, default=1.2)
    p.add_argument('--own_speed_min', type=float, default=3.0)
    p.add_argument('--own_speed_max', type=float, default=10.0)
    p.add_argument('--heading_noise_deg', type=float, default=10.0)
    # scenarios and other ship
    p.add_argument('--scenario', type=str, default='data')
    p.add_argument('--automation', type=str, default='assisted', choices=list(AUTOMATION_LEVELS) + ['agent'])
    p.add_argument('--other_policy', type=str, default='')
    p.add_argument('--other_task', type=str, default='')
    p.add_argument('--smooth_sigma', type=float, default=2.0)
    p.add_argument('--target_yaw_rate', type=float, default=0.5)
    p.add_argument('--target_lookahead', type=float, default=600.0)
    p.add_argument('--target_evasion', type=str, default=None, choices=[None, 'colreg', 'none'])
    p.add_argument('--evade_angle', type=float, default=None)
    p.add_argument('--alert_dcpa', type=float, default=None)
    p.add_argument('--alert_tcpa', type=float, default=None)
    p.add_argument('--latency', type=float, default=None)
    p.add_argument('--clear_steps', type=int, default=6)
    p.add_argument('--target_route_shape', type=str, default='straight', choices=['straight', 'data', 'auto'],
                   help='target route: straight (steady approach), the generated shape, or auto by --straighten_tol')
    p.add_argument('--straighten_tol', type=float, default=50.0,
                   help="auto: straighten the route when it deviates less than this from its chord over the whole "
                        "route [m] (50 m since 2026-09-24, mending plan part 1; was 500)")
    p.add_argument('--evasion_cooldown', type=int, default=30,
                   help='steps after an evasion before the target may alter course again (30 = 5 min)')
    # danger criteria
    p.add_argument('--d_safe', type=float, default=926.0)
    p.add_argument('--t_safe', type=float, default=900.0)
    p.add_argument('--domain_a', type=float, default=4.0)
    p.add_argument('--domain_b', type=float, default=1.6)
    p.add_argument('--collision_L', type=float, default=1.0)
    p.add_argument('--success_severity', type=int, default=2)
    p.add_argument('--hold_steps', type=int, default=2)
    # encounter lifecycle rules (ship_env_v3.EncounterRules)
    p.add_argument('--r_clear', type=float, default=2 * 1852.0, help='range above which an opening encounter counts as clearing [m]')
    p.add_argument('--n_clear', type=int, default=30, help='consecutive clearing steps (30 = 5 min) that end the encounter')
    p.add_argument('--xte_tol', type=float, default=0.1 * 1852.0, help='navigate: cross-track tolerance for "on track" [m]')
    p.add_argument('--course_tol', type=float, default=10.0, help='navigate: course tolerance for "on track" [deg]')
    p.add_argument('--recover_steps', type=int, default=18, help='navigate: on-track steps after clearing (18 = 3 min)')
    p.add_argument('--abandon_xte', type=float, default=1852.0, help='navigate: XTE beyond which the passage counts as abandoned [m]')
    p.add_argument('--abandon_steps', type=int, default=60, help='navigate: steps of abandonment (60 = 10 min) that fail')
    p.add_argument('--route_extension', type=float, default=15 * 1852.0, help='routes are extended by this much [m]')
    # environment
    p.add_argument('--decision_interval', type=float, default=10.0)
    p.add_argument('--max_ep_len', type=int, default=1080, help='safety cap only (3 h); ends as unresolved, flag 0')
    p.add_argument('--border_margin', type=float, default=3000.0)
    p.add_argument('--reward_type', type=str, default='shaped', choices=['shaped', 'final_attack_reward'])
    p.add_argument('--rudder_change_cost', type=float, default=0.05)
    p.add_argument('--continuous', type=int, default=0)
    # PPO
    p.add_argument('--num_episodes', type=int, default=3000)
    p.add_argument('--update_every_steps', type=int, default=1200, help='PPO update period in env steps')
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
    p.add_argument('--num_eval', type=int, default=24)
    args = p.parse_args(argv)
    if not args.other_task:
        args.other_task = 'navigate' if args.task == 'attack' else 'attack'
    if args.automation == 'agent' and not args.other_policy:
        p.error('--automation agent needs --other_policy <checkpoint>')
    if not args.guideline or not args.mergeline:
        gp, mp = ensure_data(args.data_dir, seed=args.seed)
        args.guideline = args.guideline or gp
        args.mergeline = args.mergeline or mp
    return args


def run(task_default='attack'):
    args = gen_args(task_default=task_default)
    if args.mode == 'train':
        train(args)
    else:
        eval_only(args)


if __name__ == '__main__':
    run('attack')
