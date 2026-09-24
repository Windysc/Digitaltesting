"""
Navigation task on the v3 encounter-lifecycle environment (counterpart of
Desktop/main_ppo.py).  Success = the encounter clears with at most a CPA
warning and the ship is back on its passage plan; failure = domain
violation / collision, passage abandoned, or leaving the map.  No
destination point, no timeout failure.

  python main_ppo_v3.py --scenario mix --num_episodes 3000
  python main_ppo_v3.py --mode eval --ckpt runs/navigate_v3/models/PPO_ShipNavigateEnv_seed0_best.pth
  python main_ppo_v3.py --mode eval --record_experiment      # scripted baseline + ShipExperiment pickle
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runner_v3 as R

if __name__ == '__main__':
    R.run('navigate')
