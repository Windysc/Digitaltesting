from gym import Env, spaces
import numpy as np
import sys
from shapely.geometry import LineString, Point
from viewer import Viewer
from ship_data import ShipExperiment
from simulator import Simulator

class ShipEnv(Env):
    def __init__(self, type='continuous', action_dim = 2, guideline_path='', mergeline_path=''):
        self.type = type
        self.action_dim = action_dim
        if type == 'continuous':
            self.action_space = 
            self.observation_space = 
            self.init_space = 
        self.ship_data = None
        self.guideline_path = guideline_path
        self.mergeline_path = guideline_path
        self.name_experiment = None
        self.last_pos = np.zeros(3)
        self.last_action = np.zeros(self.action_dim)
        self.simulator = Simulator()
        #max_x_episode = ( , )
        self.point_coming_ship = self.start_point
        self.guideline = self.load_guideline(guideline_path)
        self.mergeline = self.load_guideline(mergeline_path)
        self.start_pos = np.zeros(1)
        self.borders = self.calculate_borders(guideline_path)
        self.viewer = None
        self.test_performance = False
        self.init_test_performance = 
        self.counter = 0
    
    def side(self, x, y):
        return y-(self.point_b[1] - self.point_a[1])/(self.point_b[0] - self.point_a[0])*(x - self.point_a[0])

    def safe_distance(self, x, y):
        return np.sqrt((x - self.point_coming_ship[0])**2 + (y - self.point_coming_ship[1])**2)
    
    def load_guideline(self, npy_file):
        # Load the guideline from a .npy file
        guideline_data = np.load(npy_file)
        # Assuming the guideline data needs to be processed for tenfold intervals
        return self.process_guideline(guideline_data)
    
    def process_guideline(self, guideline_data):
        # Process the guideline data to create intervals
        tenfold_guideline = []
        for i in range(len(guideline_data) - 1):
            start = guideline_data[i]
            end = guideline_data[i + 1]
            # Create tenfold intervals between start and end
            for j in range(10):
                interpolated_point = start + (end - start) * (j / 10)
                tenfold_guideline.append(interpolated_point)
        return np.array(tenfold_guideline)

    def calculate_borders(self, guideline_data, mergeline_data):
        # Calculate the borders of the trajectory
        max_x1 = np.max(guideline_data[:, 0])
        max_y1 = np.max(guideline_data[:, 1])
        max_x2 = np.max(mergeline_data[:, 0])
        max_y2 = np.max(mergeline_data[:, 1])
        max_x = np.max(max_x1, max_x2)
        max_y = np.max(max_y1, max_y2)
        borders = np.array([
            [0, 0],
            [np.ceil(max_x)+500, 0],
            [0, np.ceil(max_y)+500],
            [np.ceil(max_x)+500, np.ceil(max_y)+500]
        ])
        return borders

    def load_guideline(self, npy_file):
        # Load the guideline from a .npy file
        guideline_data = np.load(npy_file)
        # Process the guideline data
        processed_guideline = self.process_guideline(guideline_data)
        # Calculate borders
        self.borders = self.calculate_borders(processed_guideline)
        return processed_guideline

    def side_function(self, point, line_start, line_end):
            # Determine if the point is above or below the line segment
            line = LineString([line_start, line_end])
            point = Point(point)
            return point.y - line.interpolate(line.project(point)).y

    
    def step(self, action):
        # According to the action stace a different kind of action is selected
        if self.type == 'continuous' and self.action_dim == 2:
            side = np.sign(self.side(self.last_pos[0],self.last_pos[1]))
            angle_action = action[0]*side
            rot_action = (action[1]+1)/10
        elif self.type == 'continuous' and self.action_dim == 1:
            side = np.sign(self.side(self.last_pos[0],self.last_pos[1]))
            angle_action = action[0]*side
            rot_action = 0.2
            
        state_prime = self.simulator.step(angle_level=angle_action, rot_level=rot_action)
        # convert simulator states into obervable states
        obs = self.convert_state(state_prime)
        
        # print('Observed state: ', obs)
        dn = self.end(state_prime=state_prime, obs=obs)
        rew = self.calculate_reward(obs=obs)
        self.last_pos = [state_prime[0], state_prime[1], state_prime[2]]
        self.last_action = np.array([angle_action, rot_action])
        tcpa, dcpa, cr = self.calculate_safety_method(state_prime)
        otherstates = self.point_coming_ship
        rewardmode = int(np.sqrt((self.last_pos[0]-self.point_coming_ship[0])**2+(self.last_pos[1]-self.point_coming_ship[1])**2)-300)
        if self.ship_data is not None:
            self.ship_data.new_transition(state_prime, obs, self.last_action, rew, otherstates, rewardmode, tcpa, dcpa, cr)
        info = dict()
        return obs, rew, dn, info

    def calculate_safety_method(self, state_prime):
        tcpa = self.tcpa_cal(state_prime[3], state_prime[4], state_prime[0], state_prime[1], self.point_coming_ship[0])
        dcpa = self.dcpa_cal(state_prime[3], state_prime[4], state_prime[0], state_prime[1], self.point_coming_ship[0])
        cr = self.cr_cal(tcpa, dcpa, 2000, np.sqrt(state_prime[3]**2 + state_prime[4]**2), 0.2)
        return tcpa, dcpa, cr

    def calculate_distance_to_guideline(self, position):
        # Calculate the distance from the current position to the nearest point on the guideline
        ship_point = Point((position[0], position[1]))
        d = ship_point.distance(LineString(self.guideline))  # meters
        return d
    
    def convert_state(self, state):
        """
        This method generated the features used to build the reward function
        :param state: Global state of the ship
        """
        ship_point = Point((state[0], state[1]))
        side = np.sign(self.side(state[0], state[1]))
        d = ship_point.distance(self.guideline)  # meters
        theta = state[2]  # radians
        vx = state[3]  # m/s
        vy = side*state[4]  # m/s
        thetadot = side * state[5]  # graus/min
        obs = np.array([d, theta, vx, vy, thetadot])
        return obs

    def calculate_reward(self, obs):
        d, theta, vx, vy, thetadot = obs[0], obs[1]*180/np.pi, obs[2], obs[3], obs[4]*180/np.pi
        
        # if not self.observation_space.contains(obs):
        #     return -600
        # if self.last_pos[0] > 1800 or self.last_pos[1] > 1800:
        #     return 0
        # if self.safe_distance(self.last_pos[0], self.last_pos[1]) < 80:
        #     return 1000
        # if np.sqrt((self.last_pos[0]-self.point_coming_ship[0])**2+(self.last_pos[1]-self.point_coming_ship[1])**2)>200:
        #     return 2*(1-(d/300))+0.33*(1-4/3*np.pi*abs(theta*np.pi/180-np.pi/4))+0.33*(0.5*(abs(vx)/6+abs(vy)/6))+0.33*(1-abs(thetadot)/0.4)
        # if np.sqrt((self.last_pos[0]-self.point_coming_ship[0])**2+(self.last_pos[1]-self.point_coming_ship[1])**2)<200:
        #     return 2+10*(1-np.sqrt((self.last_pos[0]-self.point_coming_ship[0])**2+(self.last_pos[1]-self.point_coming_ship[1])**2)/200)
     
    def dcpa_cal(self, vx1, vy1, x1, y1, x2):
        k = vy1 / vx1
        if k == 0:
            return 100000
        else:
            cpa_x = x1 + y1 / k
            # print("cpa_x:",cpa_x)
            t = (cpa_x - x2) / self.attacking_speed
            if t < 0:
                return 100000
            else:
                x_new = x1 + vx1 * t
                y_new = y1 + vy1 * t
                if np.sqrt((x_new - cpa_x) ** 2 + y_new ** 2) > 300:
                    return 100000
                else:
                    return np.sqrt((x_new - cpa_x) ** 2 + y_new ** 2)

    def tcpa_cal(self, vx1, vy1, x1, y1, x2):
        k = vy1 / vx1
        if k == 0:
            return 100000
        else:
            cpa_x = x1 + y1 / k
            t = (cpa_x - x2) / self.attacking_speed
            if t < 0:
                return 100000
            elif t > 300:
                return 100000
            else:
                return t

    def cr_cal(self, dcpa, tcpa, dr, v, CRal):
        return np.exp((dcpa + v * tcpa) * np.log(CRal) / dr)
   

if __name__ == '__main__':
    mode = 'train'
    if mode == 'train':
        env = ShipEnv()
        ShipExp = ShipExperiment()
        for i_episode in range(10):
            obs = env.reset()
            for t in range(10000):
                env.render()
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break
        env.close()