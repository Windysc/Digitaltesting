import numpy as np
import matplotlib.pyplot as plt
import pickle
import time as t
import datetime
import scipy.io as io
import matplotlib.ticker as mticker
from matplotlib.pyplot import MultipleLocator


class ShipExperiment:
    def __init__(self, info=None):
        """
        Initialize function to save data
        :param info: Char - Information about the iteration tets
        """
        self.iterations = -1
        self.states = {}
        self.observations = {}
        self.actions = {}
        self.rewards = {}
        self.steps = {}
        self.otherstates = {}
        self.info = info
        self.viewer = None
        self.scream = None
        self.obs_states_str = {}
        self.time_step = 10
        self.rewardmode = {}

    def new_iter(self, s0, obs0, a0, r0, other0, reward0):
        """
        A new interaction create a new sublist of states, actions and rewards, it increase the interaction count and
        initialize the steps count
        :param s0: numpy array of state 0 ie: [Xabs Yabs Thetaabs Vxabs Vyabs Thetadotabs]_0
        :param obs0: numpy array of the observation states ie: [d Vlon Theta Thetadot]_0
        :param a0: numpy array of actions ie: [Angle Propulsion]_0
        :param r0:  numpy array of reaward ie : [R]_0
        """
        self.iterations += 1
        it = self.iterations
        self.steps[it] = 0
        self.states[it] = s0
        self.observations[it] = obs0
        self.actions[it] = a0
        self.rewards[it] = r0
        self.otherstates[it] = other0
        self.rewardmode[it] = reward0

    def new_transition(self, s, obs, a, r, other, Rmode):
        """
        Each transition pass a set of numpy  arrays to be saved
        :param s: numpy array of state 0 ie: [Xabs Yabs Thetaabs Vxabs Vyabs Thetadotabs]
        :param obs: numpy array of state 0 ie: [Xabs Yabs Thetaabs Vxabs Vyabs Thetadotabs]
        :param a: numpy array of actions ie: [Angle Propulsion]_0
        :param r: numpy array of reaward ie : [R]_0
        :param other: numpy array of other states ie : [othership_x othership_y]
        """
        it = self.iterations
        self.steps[it] += 1
        self.states[it] = np.vstack([self.states[it], s])
        self.observations[it] = np.vstack([self.observations[it], obs])
        self.actions[it] = np.vstack([self.actions[it], a])
        self.rewards[it] = np.vstack([self.rewards[it], r])
        self.otherstates[it] = np.vstack([self.otherstates[it], other])
        self.rewardmode[it] = np.vstack([self.rewardmode[it], Rmode])


    def save_2mat(self, title='matlab'):
        """
        Use this method to save the vector of iteration in a Matlab format
        """
        st = datetime.datetime.fromtimestamp(t.time()).strftime('%Y%m%d%H')
        name = st+title+'.mat'
        io.savemat(name, {'states': list(self.states.values()), 'actions': list(self.actions.values()), 'obs': list(self.observations.values())})

    def save_experiment(self, descr='_experiment'):
        """
        Use this method save an experiment in .pickle format
        """
        st = datetime.datetime.fromtimestamp(t.time()).strftime('%Y-%m-%d-%H')
        name = st+descr
        with open('_experiments/'+name, 'wb') as f:
            pickle.dump(self.__dict__, f, 2)
        f.close()

    def load_from_experiment(self, name):
        """
        Use this method save an experiment in .pickle format
        :param name:
        """
        with open('_experiments/' + name, 'rb') as f:
            tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)

    def plot_actions(self, iter=0, time=True):
        """
        Plot actions of an iteration
        :param iter: iteration index
        """
        title = {0:'Rudder action', 1:'Propulsion action'}
        if iter == -1:
            f, axarr = plt.subplots(len(self.actions[0][0, :]), sharex=True)
            for j in range(self.iterations+1):
                for i in range(len(self.actions[0][0, :])):
                    if time:
                        axarr[i].plot(self.time_step*np.arange(0, self.steps[j], 1), self.actions[j][1:, i],  label="k="+str(j))
                        axarr[i].set_xlabel('time (s)')
                    else:
                        axarr[i].plot(np.arange(0, self.steps[j], 1), self.actions[j][1:, i],  label="k="+str(j))
                        axarr[i].set_xlabel('steps')
                    axarr[i].set_title(title[i])
                    axarr[i].set_ylabel('Actions')
                plt.legend(loc='right', bbox_to_anchor=(0.7, 1.1, 0.5, 1.1), borderaxespad=0.)
        else:
            f, axarr = plt.subplots(len(self.actions[iter][0, :]), sharex=True)
            for i in range(len(self.actions[iter][0, :])):
                if time:
                    axarr[i].plot(self.time_step*np.arange(0, self.steps[iter], 1), self.actions[iter][1:, i])
                    axarr[i].set_xlabel('time (s)')
                else:
                    axarr[i].plot(np.arange(0, self.steps[iter], 1), self.actions[iter][1:, i])
                    axarr[i].set_xlabel('steps')
                axarr[i].set_title(title[i])
                axarr[i].set_ylabel('Actions')

        for a in axarr.flatten():
            a.xaxis.set_tick_params(labelbottom=True)
            for tk in a.get_yticklabels():
                tk.set_visible(True)
            for tk in a.get_xticklabels():
                tk.set_visible(True)
        plt.show()

    def plot_reward(self, iter=0):
        """
        Plot reward of an iteration
        :param iter: iteration index
        """
        if iter == -1:
            for i in range(self.iterations+1):
                plt.plot(np.arange(0, self.steps[i] + 1, 1), self.rewards[i])
                plt.ylabel('Reward')
                plt.xlabel('steps')
                plt.show()
        else:
            plt.plot(np.arange(0, self.steps[iter]+1, 1), self.rewards[iter])
            plt.ylabel('Reward')
            plt.xlabel('steps')
            plt.show()

    def plot_tcpa(self,iter):
        if iter !=-1:
            for j in range(iter+1):
                plt.plot(np.arange(0, self.steps[j], 1), self.tcpa[j][1:, 0],label="k="+str(j))
            plt.ylabel('TCPA')
            plt.xlabel('steps')
            plt.show()

    def plot_dcpa(self,iter):
        if iter!=-1:
            for j in range(iter+1):
                plt.plot(np.arange(0, self.steps[j], 1), self.dcpa[j][1:, 0],label="k="+str(j))
            plt.ylabel('DCPA')
            plt.xlabel('steps')
            plt.show()

    def plot_CR(self, iter):
        if iter != -1:
            for j in range(iter + 1):
                plt.plot(np.arange(0, self.steps[j], 1), self.cr[j][1:, 0], label="k=" + str(j))
            plt.ylabel('CR')
            plt.xlabel('steps')
            plt.show()


    def plot_obs(self, iter=0, time=True):
        img, ax = plt.subplots(5, sharex=True)
        self.obs_states_str[0] = 'd'
        self.obs_states_str[1] = 'Θ'
        self.obs_states_str[2] = 'vx'
        self.obs_states_str[3] = 'vy'
        self.obs_states_str[4] = 'dΘ/dt'
        if iter == -1:
            for j in range(self.iterations+1):
                ax[0].set_title("Observed states")
                for i in range(5):
                    ax[i].set_ylabel(self.obs_states_str[i])
                    if time:
                        ax[i].plot(self.time_step*np.arange(0, self.steps[j], 1), self.observations[j][1:, i], label="k="+str(j))
                        ax[i].set_xlabel('time (s)')
                    else:
                        ax[i].plot(np.arange(0, self.steps[j], 1), self.observations[j][1:, i], label="k="+str(j))
                        ax[i].set_xlabel('steps')
                        formatter = mticker.ScalarFormatter()
                        ax[i].xaxis.set_major_formatter(formatter)
            plt.legend(loc='right', bbox_to_anchor=(0.7, 2.8, 0.5, 2.8), borderaxespad=0.)
        else:
            for i in range(5):
                ax[i].set_title("Observed states")
                ax[i].set_ylabel("Obs" + str(i))
                if time:
                    ax[i].plot(np.arange(0, self.time_step*self.steps[iter], 1), self.observations[iter][1:, i])
                    ax[i].set_xlabel('time (s)')
                else:
                    ax[i].plot(np.arange(0, self.steps[iter], 1), self.observations[iter][1:, i])
                    ax[i].set_xlabel('steps')
        for a in ax.flatten():
            a.xaxis.set_tick_params(labelbottom=True)
            for tk in a.get_yticklabels():
                tk.set_visible(True)
            for tk in a.get_xticklabels():
                tk.set_visible(True)
        plt.show()


    def show_doubled_lane(polygon):
        """
        args: ndarray in shape of (n, 2)
        returns:
        """
        xs, ys = polygon[:, 0], polygon[:, 1]
        plt.plot(xs, ys, '--', color='grey')


    def plot_trajectory(self, iter=0):
        """
        Plot trajectory of an iteration
        :param iter: iteration index
        """
        plt.axis('equal')
        plt.axis([0, 2000, 0, 2000])
        x_major_locator = MultipleLocator(500)
        y_major_locator = MultipleLocator(500)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)
        plt.plot()
        if iter == -1:
            for i in range(self.iterations + 1):
                for k in (range(len(self.rewardmode[i]))):
                    if (self.rewardmode[i][k]>0):
                            line1,=plt.plot(self.states[i][k][0], self.states[i][k][1], linestyle='-', linewidth=1, marker='.', markersize=0.8, color='r')
                            line2,=plt.plot(self.otherstates[i][k][0], self.otherstates[i][k][1], linestyle='-', linewidth=1, marker='.', markersize=0.8, color='b')

                    else:
                        if ((self.rewardmode[i][k-1]>0)&(self.rewardmode[i][k]<0)):
                            circle1=plt.Circle(xy=((self.states[i][k][0]+self.otherstates[i][k][0])/2, (self.states[i][k][1]+self.otherstates[i][k][1])/2), fill=False, radius=150, linewidth=1, linestyle='--', edgecolor='0.2')
                            plt.gca().add_artist(circle1)

                        if (self.rewardmode[i][k] < 0):
                            plt.plot(self.states[i][k][0], self.states[i][k][1], 'o:g', linewidth=1, markersize=0.8)
                            plt.plot(self.otherstates[i][k][0], self.otherstates[i][k][1], 'o:b', linewidth=1, markersize=0.8)

                circle2 = plt.Circle((self.states[i][len(self.rewardmode[i])-1][0], self.states[i][len(self.rewardmode[i])-1][1]), fill=False, radius=50, linewidth=1, linestyle='--', edgecolor='0')
                circle3 = plt.Circle((self.otherstates[i][len(self.rewardmode[i])-1][0], self.otherstates[i][len(self.rewardmode[i])-1][1]), fill=False, radius=50, linewidth=1, linestyle='--', edgecolor='0.5')
                plt.gca().add_artist(circle2)
                plt.gca().add_artist(circle3)

            plt.legend([line1, line2], ['Vessel', 'Objective Vessel'], loc='upper right')
            plt.ylabel('Y axis(meters)')
            plt.xlabel('X axis(meters)')
            plt.show()

        else:
            plt.axis('equal')
            plt.axis([0, 2000, 0, 2000])
            for k in (range(len(self.rewardmode[iter]))):
                if (self.rewardmode[iter][k] > 0):
                    plt.plot(self.states[iter][k][0], self.states[iter][k][1], 'o:r', linewidth=1, markersize=0.8)
                    plt.plot(self.otherstates[iter][k][0], self.otherstates[iter][k][1], 'o:b', linewidth=1, markersize=0.8)

                else:
                    if (self.rewardmode[iter][k] == 0):
                        plt.Circle(((self.states[iter][k][0]+self.otherstates[iter][k][0])/2, (self.states[iter][k][1]+self.otherstates[iter][k][1]/2)), 30, color='g')

                    if (self.rewardmode[iter][k] < 0):
                        plt.plot(self.states[iter][k][0], self.states[iter][k][1], 'o:g', linewidth=1, markersize=0.8)
                        plt.plot(self.otherstates[iter][k][0], self.otherstates[iter][k][1], 'o:b', linewidth=1, markersize=0.8)

            plt.Circle((self.states[iter][len(self.rewardmode[iter])-1][0], self.states[iter][len(self.rewardmode[iter])-1][1]),30, color='r')
            plt.Circle((self.otherstates[iter][len(self.rewardmode[iter])-1][0], self.otherstates[iter][len(self.rewardmode[iter])-1]), 30, color='b')
            plt.legend(['Vessel', 'Objective vessel'])
            plt.ylabel('Y')
            plt.xlabel('X')
            plt.show()
            
    def plot_reward_change(self, iter):
        """
        Plot reward of an iteration
        :param iter: iteration index
        """
        if iter == -1:
            for i in range(self.iterations+1):
                plt.plot(np.arange(0, self.steps[i]+1, 1), self.rewards[i])
                plt.ylabel('Reward')
                plt.xlabel('steps')
                plt.show()
        else:
            plt.plot(np.arange(0, self.steps[iter]+1, 1), self.rewards[iter])
            plt.ylabel('Reward')
            plt.xlabel('steps')
            plt.show()


    def compute_settling_time_d(self, iter=0):
        d = self.observations[iter][:, 0]
        for j in reversed(range(len(d)-10)):
            if d[j] > 18:
                return (j + 1)* 10
        return len(d)*10

    def compute_settling_time_v(self, iter=0):
        v = self.observations[iter][:, 2]
        for j in reversed(range(len(v))):
            if v[j] < 1.8:
                return (len(v)-(j + 1)) * 10
        return len(v)*10

    def plot_settling_time(self):
        st_d = np.zeros(self.iterations+1)
        st_v = np.zeros(self.iterations + 1)
        for i in range(self.iterations+1):
            st_d[i] = self.compute_settling_time_d(i)
            st_v[i] = self.compute_settling_time_v(i)
        plt.title('Settling time ')
        plt.xlabel('Episode')
        plt.ylabel('time (s)')
        plt.plot(st_d, 'o-', label='d')
        plt.plot(st_v, 'o-',  label='vx')
        plt.legend()
