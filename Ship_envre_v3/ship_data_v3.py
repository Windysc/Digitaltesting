"""
Episode recorder in the style of SimpleShipAI/ship_data.py (ShipExperiment),
extended the way Ship_envre/ship_data_vis.py did it: besides states,
observations, actions and rewards it keeps the other ship's state, the
encounter phase (in place of the old `rewardmode`) and the route.  Saved as
a pickle of the object's __dict__ under `_experiments/`, so the plotting
helpers of the original repository can load it with load_from_experiment().

    rec = ShipExperiment(info='attack eval')
    env.set_recorder(rec)
    ... run episodes ...
    rec.save_experiment('attack_eval')
"""
import os
import pickle
import datetime
import numpy as np


class ShipExperiment:
    def __init__(self, info=None, time_step=10.0):
        self.iterations = -1
        self.states, self.observations, self.actions, self.rewards = {}, {}, {}, {}
        self.otherstates, self.phase, self.guideline, self.steps = {}, {}, {}, {}
        self.info = info
        self.time_step = time_step

    def new_iter(self, s0, obs0, a0, r0, other0, phase0, g0):
        self.iterations += 1
        it = self.iterations
        self.steps[it] = 0
        self.states[it] = np.asarray(s0, float)[None, :]
        self.observations[it] = np.asarray(obs0, float)[None, :]
        self.actions[it] = np.asarray(a0, float)[None, :]
        self.rewards[it] = np.asarray(r0, float).reshape(1, -1)
        self.otherstates[it] = np.asarray(other0, float)[None, :]
        self.phase[it] = np.array([[phase0]], float)
        self.guideline[it] = np.asarray(g0, float)

    def new_transition(self, s, obs, a, r, other, phase, g0=None):
        it = self.iterations
        self.steps[it] += 1
        self.states[it] = np.vstack([self.states[it], np.asarray(s, float)])
        self.observations[it] = np.vstack([self.observations[it], np.asarray(obs, float)])
        self.actions[it] = np.vstack([self.actions[it], np.asarray(a, float)])
        self.rewards[it] = np.vstack([self.rewards[it], np.asarray(r, float).reshape(1, -1)])
        self.otherstates[it] = np.vstack([self.otherstates[it], np.asarray(other, float)])
        self.phase[it] = np.vstack([self.phase[it], [[phase]]])

    def _dir(self):
        os.makedirs('_experiments', exist_ok=True)
        return '_experiments'

    def save_experiment(self, descr='_experiment'):
        st = datetime.datetime.now().strftime('%Y-%m-%d-%H')
        name = '%s%s' % (st, descr)
        with open(os.path.join(self._dir(), name), 'wb') as f:
            pickle.dump(self.__dict__, f, 2)
        print('experiment saved: _experiments/%s (%d episodes)' % (name, self.iterations + 1))
        return name

    def load_from_experiment(self, name):
        with open(os.path.join('_experiments', name), 'rb') as f:
            self.__dict__.update(pickle.load(f))
        return self

    def save_2mat(self, title='matlab'):
        import scipy.io as io
        name = datetime.datetime.now().strftime('%Y%m%d%H') + title + '.mat'
        io.savemat(name, {'states': list(self.states.values()), 'actions': list(self.actions.values()),
                          'obs': list(self.observations.values()), 'other': list(self.otherstates.values()),
                          'phase': list(self.phase.values())})
        return name

    # ---- quick plots (matplotlib, headless-safe) ----------------------------
    def plot_trajectory(self, iters=-1, save_path=None):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        its = range(self.iterations + 1) if iters == -1 else [iters]
        fig, ax = plt.subplots(figsize=(7, 7))
        for it in its:
            s, o = self.states[it], self.otherstates[it]
            ax.plot(s[:, 0], s[:, 1], 'b-', lw=1)
            ax.plot(o[:, 0], o[:, 1], 'r-', lw=1)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
        ax.set_title('%s: %d episodes' % (self.info, len(list(its))))
        if save_path:
            fig.savefig(save_path, dpi=110, bbox_inches='tight')
        plt.close(fig)

    def plot_actions(self, it=0, save_path=None):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        a = self.actions[it]
        t = np.arange(len(a)) * self.time_step
        fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        ax[0].step(t, a[:, 0], where='post'); ax[0].set_ylabel('rudder level')
        ax[1].step(t, a[:, 1], where='post'); ax[1].set_ylabel('throttle'); ax[1].set_xlabel('time [s]')
        for x in ax:
            x.grid(True, alpha=0.3)
        if save_path:
            fig.savefig(save_path, dpi=110, bbox_inches='tight')
        plt.close(fig)
