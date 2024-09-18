from gym import Env, spaces
import numpy as np
import sys
from shapely.geometry import LineString, Point
from viewer import Viewer
from ship_data import ShipExperiment
from simulator import Simulator

class ShipEnv(Env):
    def __init__(self, type='continuous', action_dim = 2, guideline_path='/home/junze/.jupyter/Digitaltesting/Ship_envre/datasettest_1.npy', mergeline_path='/home/junze/.jupyter/Digitaltesting/Ship_envre/datasettest_1.npy'):
        self.type = type
        self.action_dim = action_dim
        if type == 'continuous':
            self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]))
            self.observation_space = spaces.Box(low=np.array([0, -np.pi, -5.0, -5.0, -2.0]), high=np.array([150, np.pi, 5.0, 5.0, 2.0]))
            self.init_space = spaces.Box(low=np.array([0, -0.2, -5.0, -5.0, -2.0]), high=np.array([150, 0.2, 5.0, 5.0, 2.0]))
        self.ship_data = None
        self.guideline = self.load_guideline(guideline_path)
        self.mergeline = self.load_guideline(guideline_path)
        self.name_experiment = None
        self.last_pos = np.zeros(3)
        self.last_action = np.zeros(self.action_dim)
        self.simulator = Simulator()
        self.guideline = self.convert_lat_lon_to_meters(self.guideline)
        self.mergeline = self.convert_lat_lon_to_meters(self.mergeline)
        self.point_coming_ship = self.mergeline[0][0]
        self.processed_guideline = self.process_guideline(self.guideline)
        self.start_pos = self.guideline[0][0]
        self.borders = self.calculate_borders(self.guideline, self.mergeline)
        self.viewer = None
        self.test_performance = False
        self.init_test_performance = (np.linspace(-2.2, -2.4, 10))
        self.counter = 0
        self.single_step = 0
        self.attacking_speed = 3   
        
    def load_guideline(self, npy_file):
        # Load the guideline from a .npy file
        data = np.load(npy_file, allow_pickle=True)  # Return the raw data for further processing

        if isinstance(data, np.ndarray):
            # If it's a 3D array, reshape it to 2D
            if data.ndim == 3 and data.shape[2] == 2:
                data = data.reshape(-1, 2)
            else:
                raise ValueError(f"Data must be a 3D array with shape (n, m, 2). Got shape: {data.shape}")
        else:
            raise ValueError("Data must be a numpy array")

        return data
    
    def convert_lat_lon_to_meters(self, lat_lon_data):
        def haversine(lat1, lon1, lat2, lon2):
                R = 6371000  # Earth radius in meters
                phi1 = np.radians(lat1)
                phi2 = np.radians(lat2)
                delta_phi = np.radians(lat2 - lat1)
                delta_lambda = np.radians(lon2 - lon1)

                a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
                c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

                return R * c

            # Ensure lat_lon_data is a list of lists
        if isinstance(lat_lon_data, np.ndarray):
                lat_lon_data = lat_lon_data.tolist()

        if not isinstance(lat_lon_data, list):
                raise ValueError("Input data must be a list of lists")

        meters_data = []

        for segment in lat_lon_data:
            if not isinstance(segment, list):
                raise ValueError("Each segment must be a list of points")

            segment_meters = []
            for point in segment:
                if isinstance(point, (list, tuple)) and len(point) == 2:
                    segment_meters.append([point[0], point[1]])  # Directly append the points
            else:
                raise ValueError("Each point in a segment must be a list or tuple of two elements (longitude, latitude)")

        if segment_meters:
                meters_data.append(np.array(segment_meters))

        return meters_data

    def safe_distance(self, x, y):
        return np.sqrt((x - self.point_coming_ship[0])**2 + (y - self.point_coming_ship[1])**2)

    def process_guideline(self, guideline):
        # Process the guideline data to group every ten points into a different row
        tenfold_guideline = []
        for i in range(0, len(guideline), 10):
            tenfold_guideline.append(guideline[i:i+10])
        return np.array(tenfold_guideline, dtype=object)

    def calculate_trajectory_statistics(self, data_meters):
        # Calculate statistics for each time stamp
        avg_trajectory = np.mean(data_meters, axis=0)
        std_trajectory = np.std(data_meters, axis=0)

        # Percentile-based selection
        lower_bound_percentile = np.percentile(data_meters, 2.5, axis=0)
        upper_bound_percentile = np.percentile(data_meters, 97.5, axis=0)

        # 3-standard-deviation selection
        lower_bound_std = avg_trajectory - 3 * std_trajectory
        upper_bound_std = avg_trajectory + 3 * std_trajectory

        # Select trajectories closest to the bounds
        def find_closest_trajectory(trajectories, target):
            distances = np.sum(np.sqrt(np.sum((trajectories - target)**2, axis=2)), axis=1)
            return trajectories[np.argmin(distances)]

        lower_trajectory_percentile = find_closest_trajectory(data_meters, lower_bound_percentile)
        upper_trajectory_percentile = find_closest_trajectory(data_meters, upper_bound_percentile)
        lower_trajectory_std = find_closest_trajectory(data_meters, lower_bound_std)
        upper_trajectory_std = find_closest_trajectory(data_meters, upper_bound_std)

        return {
            "avg_trajectory": avg_trajectory,
            "lower_trajectory_percentile": lower_trajectory_percentile,
            "upper_trajectory_percentile": upper_trajectory_percentile,
            "lower_trajectory_std": lower_trajectory_std,
            "upper_trajectory_std": upper_trajectory_std
        }

    def calculate_borders(self, guideline, mergeline):
        def process_ragged_array(arr):
            if isinstance(arr, list):
                # Flatten the list of arrays
                return np.concatenate([np.array(subarr) for subarr in arr])
            elif isinstance(arr, np.ndarray) and arr.dtype == object:
                # Flatten the numpy array of arrays
                return np.concatenate([subarr for subarr in arr])
            else:
                return np.array(arr)

        # Process the inputs
        guideline = process_ragged_array(guideline)
        mergeline = process_ragged_array(mergeline)

        # Ensure the arrays are 2D
        if guideline.ndim != 2 or mergeline.ndim != 2:
            raise ValueError("Processed input arrays must be 2D")

        # Calculate the borders of the trajectory
        max_x1 = np.max(guideline[:, 0])
        max_y1 = np.max(guideline[:, 1])
        max_x2 = np.max(mergeline[:, 0])
        max_y2 = np.max(mergeline[:, 1])
        max_x = max(max_x1, max_x2)
        max_y = max(max_y1, max_y2)

        borders = np.array([
            [0, 0],
            [np.ceil(max_x)+500, 0],
            [0, np.ceil(max_y)+500],
            [np.ceil(max_x)+500, np.ceil(max_y)+500]
        ])

        return borders



    def step(self, action):
        # According to the action space a different kind of action is selected
        if self.type == 'continuous' and self.action_dim == 2:
            angle_action = action[0]
            rot_action = (action[1] + 1) / 10
                     
        state_prime = self.simulator.step(angle_level=angle_action, rot_level=rot_action)
        # convert simulator states into observable states
        obs = self.convert_state(state_prime)
        # print('Observed state: ', obs)
        dn = self.end(state_prime=state_prime, obs=obs)
        rew = self.calculate_reward(obs=obs)
        self.point_coming_ship = [self.mergeline[self.single_step][0][0], self.mergeline[self.single_step][0][1]]
        self.single_step += 1
        self.last_pos = [state_prime[0], state_prime[1], state_prime[2]]
        self.last_action = np.array([angle_action, rot_action])
        tcpa, dcpa, cr = self.calculate_safety_method(state_prime)
        otherstates = self.point_coming_ship
        rewardmode = int(np.sqrt((self.last_pos[0] - self.point_coming_ship[0])**2 + (self.last_pos[1] - self.point_coming_ship[1])**2) - 150)
        if self.ship_data is not None:
            self.ship_data.new_transition(state_prime, obs, self.last_action, rew, otherstates, rewardmode, tcpa, dcpa, cr)
        info = dict()
        return obs, rew, dn, info


    def calculate_safety_method(self, state_prime):
        tcpa = self.tcpa_cal(state_prime[3], state_prime[4], state_prime[0], state_prime[1], self.point_coming_ship[0])
        dcpa = self.dcpa_cal(state_prime[3], state_prime[4], state_prime[0], state_prime[1], self.point_coming_ship[0])
        cr = self.cr_cal(tcpa, dcpa, 2000, np.sqrt(state_prime[3]**2 + state_prime[4]**2), 0.2)
        return tcpa, dcpa, cr

    def calculate_distance_to_guideline(self, point):
        ship_point = point
        # Use the tenfold guideline data at the current time step
        step_index = self.single_step // 10  # Ensure the index is an integer
        d = ship_point.distance(LineString(self.guideline[step_index]))  
        return d
    
    def convert_state(self, state):
        """
        This method generated the features used to build the reward function
        :param state: Global state of the ship
        """
        ship_point = Point((state[0], state[1]))
        d = self.calculate_distance_to_guideline(ship_point)  # meters
        theta = state[2]  # radians
        vx = state[3]  # m/s
        vy = state[4]  # m/s
        thetadot = state[5]  # graus/min
        obs = np.array([d, theta, vx, vy, thetadot], dtype=object)
        return obs

    def calculate_reward(self, obs):
        d, theta, vx, vy, thetadot = obs[0], obs[1]*180/np.pi, obs[2], obs[3], obs[4]*180/np.pi
        if self.last_pos[0] > 1800 or self.last_pos[1] > 1800:
            return 0
        if self.safe_distance(self.last_pos[0], self.last_pos[1]) < 80:
            return 1000
        if np.sqrt((self.last_pos[0]-self.point_coming_ship[0])**2+(self.last_pos[1]-self.point_coming_ship[1])**2)>200:
            return 2*(1-(d/300))+0.33*(1-4/3*np.pi*abs(theta*np.pi/180-np.pi/4))+0.33*(0.5*(abs(vx)/6+abs(vy)/6))+0.33*(1-abs(thetadot)/0.4)
        if np.sqrt((self.last_pos[0]-self.point_coming_ship[0])**2+(self.last_pos[1]-self.point_coming_ship[1])**2)<200:
            return 2+10*(1-np.sqrt((self.last_pos[0]-self.point_coming_ship[0])**2+(self.last_pos[1]-self.point_coming_ship[1])**2)/200)
        
    def end(self, state_prime, obs):
        if not self.observation_space.contains(obs) or self.safe_distance(self.last_pos[0], self.last_pos[1]) < 80:
            if not self.observation_space.contains(obs):
                print("\n Smashed")
                print(obs)
            if self.safe_distance(self.last_pos[0], self.last_pos[1]) < 80:
                print("\n Too close and generation complete")
            if self.viewer is not None:
                self.viewer.end_episode()
            if self.ship_data is not None:
                if self.ship_data.iterations > 0:
                    self.ship_data.save_experiment(self.name_experiment)
            return True
        else:
            return False
        
    def set_init_space(self, low, high):
        self.init_space = spaces.Box(low=np.array(low), high=np.array(high))
     
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
    
    def reset(self):
        init = list(map(float, self.init_space.sample()))
        if self.test_performance:
            angle = self.init_test_performance[self.counter]
            v_lon = 3
            init_states = np.array([self.start_pos[0], 5, angle, v_lon * np.cos(angle), v_lon * np.sin(angle), 0])
            self.counter += 1
            init[0] = 0
            init[1] = angle
        else:
            init_states = np.array([self.start_pos[0], init[0], init[1], init[2] * np.cos(init[1]), init[2] * np.sin(init[1]), 0])
        self.simulator.reset_start_pos(init_states)
        self.last_pos = np.array([self.start_pos[0], init[0],  init[1]])
        self.point_coming_ship = [self.mergeline[0][0][0], self.mergeline[0][0][1]]
        state1 = self.simulator.get_state()
        print('Reseting position')
        state = self.simulator.get_state()
        tcpa = self.tcpa_cal(state1[3], state1[4], state1[0], state1[1], self.point_coming_ship[0])
        dcpa = self.dcpa_cal(state1[3], state1[4], state1[0], state1[1], self.point_coming_ship[0])
        cr = self.cr_cal(dcpa, tcpa, 2000, np.sqrt(state1[3] ** 2 + state1[4] ** 2), 0.3)
        otherstates = self.point_coming_ship
        rewardmode = int(np.sqrt((self.last_pos[0]-self.point_coming_ship[0])**2+(self.last_pos[1]-self.point_coming_ship[1])**2)-300)
        if self.ship_data is not None:
            if self.ship_data.iterations > 0:
                self.ship_data.save_experiment(self.name_experiment)
            self.ship_data.new_iter(state, self.convert_state(state), np.zeros(len(self.last_action)), np.array([0]), otherstates, rewardmode, LineString([self.point_a, self.point_b]), tcpa, dcpa, cr)
        if self.viewer is not None:
            self.viewer.end_episode()
        return self.convert_state(state)


if __name__ == '__main__':
    mode = 'test'  
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
    elif mode == 'test':
        env = ShipEnv()
        obs = env.reset()
        for t in range(50):
            action = env.action_space.sample()  
            obs, reward, done, info = env.step(action)
            print(f"Step {t + 1}: obs={obs}, reward={reward}, done={done}, info={info}")
            if done:
                print("Test finished after {} timesteps".format(t + 1))
                break
        env.close()