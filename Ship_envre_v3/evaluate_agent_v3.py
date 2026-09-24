"""
Scorecard (scenario families x automation levels, event grades and scores)
for a v3 checkpoint or the scripted baseline: the v2 tool driven by runner_v3.

  python evaluate_agent_v3.py --task attack --ckpt checkpoints/ppo_attack_v3_best.pth --n 5 --out scorecards/attack_v3
  python evaluate_agent_v3.py --task navigate --n 5 --out scorecards/navigate_v3_baseline
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(os.path.dirname(HERE), 'Ship_envre_v2')
for p_ in (HERE, V2):
    if p_ not in sys.path:
        sys.path.insert(0, p_)

import runner_v3 as R              # noqa: E402
import evaluate_agent as E         # noqa: E402  (Ship_envre_v2, unchanged on disk)
E.M = R

if __name__ == '__main__':
    E.main()
