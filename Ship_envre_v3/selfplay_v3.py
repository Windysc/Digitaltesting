"""
Self-play on the v3 environment: attack agent vs navigation policy, then
navigation agent vs attack policy, per round (the v2 script driven by
runner_v3).

  python selfplay_v3.py --rounds 3 --out runs/selfplay_v3 --scenario mix --num_episodes 600
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(os.path.dirname(HERE), 'Ship_envre_v2')
for p_ in (HERE, V2):
    if p_ not in sys.path:
        sys.path.insert(0, p_)

import runner_v3 as R          # noqa: E402
import selfplay as S           # noqa: E402  (Ship_envre_v2, unchanged on disk)
S.M = R

if __name__ == '__main__':
    S.main()
