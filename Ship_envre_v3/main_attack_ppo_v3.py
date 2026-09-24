"""
Attack task on the v3 encounter-lifecycle environment (counterpart of
Desktop/main_attack_ppo.py).  Success = a dangerous encounter by maritime
criteria; failure = the encounter clears without it; no destination point,
no timeout failure.

  python main_attack_ppo_v3.py --scenario mix --num_episodes 3000
  python main_attack_ppo_v3.py --mode eval --ckpt runs/attack_v3/models/PPO_ShipAttackEnv_seed0_best.pth
  python main_attack_ppo_v3.py --mode eval --render          # scripted baseline in the live viewer
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runner_v3 as R

if __name__ == '__main__':
    R.run('attack')
