"""
Regenerated counterpart of Desktop/main_ppo.py: navigation task.

The own ship must follow its generated guideline route to the destination
(end of the route, within --dest_radius and --dest_heading_tol, like the
navigation_target of the reference) while the other ship travels its own
generated route as a moving hazard (the reference's moving obstacle).  The
episode fails as soon as the encounter severity reaches 2 (ship-domain
violation) or 3 (collision); a CPA warning (severity 1) is penalised but not
terminal.  Success rate, logs, checkpoints and scene figures follow
main_ppo.py; everything is shared with main_attack_ppo_ship.py, only the
task and the output directory differ.

Usage
-----
  python main_ppo_ship.py --num_episodes 3000                       # train on synthetic routes
  python main_ppo_ship.py --guideline <..>.npy --mergeline <..>.npy  # train on generated data
  python main_ppo_ship.py --mode eval --ckpt runs/navigate_v2/models/PPO_ShipNavigateEnv_seed0_best.pth
  python main_ppo_ship.py --mode eval                                # scripted guideline-following baseline
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import main_attack_ppo_ship as M

if __name__ == '__main__':
    M.run('navigate')
