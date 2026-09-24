"""
gym.py -- compatibility shim so the unmodified reference scripts can `import gym`.

The legacy `gym` package is unmaintained and not installed here (Python 3.14,
numpy 2.x); `gymnasium` is its maintained successor with the same API.  This
file sits next to main_ppo.py / main_attack_ppo.py and re-exports gymnasium.
Scripts run from the Desktop therefore see gymnasium under the name `gym`.
"""
import gymnasium as _gymnasium
from gymnasium import *  # noqa: F401,F403
from gymnasium import Env, spaces, make  # noqa: F401

__version__ = getattr(_gymnasium, '__version__', 'gymnasium')
