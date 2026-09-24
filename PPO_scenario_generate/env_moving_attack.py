"""
env_moving_attack.py -- MassTestingEnv for Desktop/main_attack_ppo.py (attack task).

Rebuilt 2026-09-15; shares the world model of env_moving_obj.py.  The own
ship is the attacker.  Targets are the ships in `ts_list`, or, when that list
is empty (as in main_attack_ppo.py), every moving obstacle.  Static obstacles
stay hazards.

Success (flag 1): the own ship comes within `attack_range` of a target hull
(default one target ship length, a close-quarters criterion).  Failure
(flag -1): collision with a hazard or leaving the map.  Timeout: flag 0.

reward_type 'final_attack_reward' (script default): terminal only, +10 on
success, -10 on failure, and at timeout -10 x (closest approach / initial
distance).  'dense_attack_reward' adds per-decision closing progress and a
bonus while inside the target's risk range.
"""
from env_moving_obj import (MassTestingEnv as _BaseEnv, ownship, navigation_target, obstacle,
                            ACTION_TABLE, ACTION_NAMES)


class MassTestingEnv(_BaseEnv):
    def __init__(self, own_ship, ts_list, ob_list, nt, duration=60000, decision_interval=600,
                 reward_type='final_attack_reward', X_LEN=2000, Y_LEN=1000, save_dir='.',
                 attack_range=None, init_noise=0.0):
        super().__init__(own_ship, ts_list, ob_list, nt, duration=duration,
                         decision_interval=decision_interval, reward_type=reward_type,
                         X_LEN=X_LEN, Y_LEN=Y_LEN, save_dir=save_dir, task='attack',
                         attack_range=attack_range, init_noise=init_noise)


__all__ = ['MassTestingEnv', 'ownship', 'navigation_target', 'obstacle', 'ACTION_TABLE', 'ACTION_NAMES']
