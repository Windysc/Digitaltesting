import numpy as np
import pickle
import matplotlib.pyplot as plt
from ship_data import ShipExperiment
import matplotlib.animation as animation

filename = 'ddpg__redetorcsTESTN6r_kit_rn4_maior2_mem20k_target01_theta3_batch32_adam2'
with open('_experiments/history_' + filename+'.pickle', 'rb') as f:
    hist = pickle.load(f)
f.close()

def _moving_average(a, n=20) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

plt.figure()
plt.title('Reward Evolution')
plt.xlabel('Episodes')
plt.ylabel('Reward')
rw = _moving_average(hist['episode_reward'])
plt.plot(rw)

plt.figure()
plt.title('Survival Evolution')
plt.xlabel('Episode')
plt.ylabel('Steps in the episode')
nsteps = _moving_average(hist['nb_episode_steps'])
plt.plot(nsteps)
plt.show()

# Here you can load and plot you performance test
shipExp = ShipExperiment()
experiment_name = '2024-07-11-07experiment_ssn_ddpg_10iter'
shipExp.load_from_experiment(experiment_name)
shipExp.plot_obs(iter=-1) # seleciona os episodios manualmente ou coloque -1 para plotar todos
shipExp.plot_settling_time()
shipExp.plot_actions(iter=9)
shipExp.plot_trajectory([0,0], [1200, 1200], iter=-1)
shipExp.plot_reward_change(iter=1)
shipExp.plot_dcpa(iter=9)
shipExp.plot_tcpa(iter=9)
shipExp.plot_CR(iter=9)
shipExp.plot_trajectory_animation()
