"""
Animations for the v3 environment: the v2 renderer (plain full-scope map +
detailed display + panels, storyboards, grid GIFs) driven by runner_v3.

  python animate_v3.py --ckpt checkpoints/ppo_attack_v3_mix_best.pth --scenario mix --episodes 3 --grid 6
  python animate_v3.py --task navigate --episodes 2                 # scripted navigation baseline
Extra arguments go to runner_v3.gen_args.

The plain full-scope map keeps a fixed extent for the whole episode; in v3
that extent is taken from the two ships' tracks only (the routes are
extended 15 nm and would otherwise dominate the view).
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(os.path.dirname(HERE), 'Ship_envre_v2')
for p_ in (HERE, V2):
    if p_ not in sys.path:
        sys.path.insert(0, p_)

import runner_v3 as R            # noqa: E402
import animate_scenes as A       # noqa: E402  (Ship_envre_v2, unchanged on disk)
A.M = R                          # point the renderer at the v3 runner / environment

NM = 1852.0


def full_extent_v3(rec, margin=0.15):
    """Fixed square extent covering both ships' whole-episode tracks."""
    pts = np.concatenate([rec['state'][:, :2], rec['target']])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    c = 0.5 * (lo + hi)
    half = max(0.5 * max(hi - lo) * (1 + 2 * margin), 2 * NM)
    return np.array([[c[0] - half, c[1] - half], [c[0] + half, c[1] - half],
                     [c[0] - half, c[1] + half], [c[0] + half, c[1] + half]])


A.full_extent = full_extent_v3

if __name__ == '__main__':
    A.main()
