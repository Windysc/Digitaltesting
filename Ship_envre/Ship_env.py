from gym import Env, spaces
import numpy as np
import sys
from shapely.geometry import LineString, Point
from viewer import Viewer
from ship_data_vis import ShipExperiment
from simulator import Simulator

class ShipEnv(Env):
    def __init__(self, type='L1 movement', action_dim=2,
                 guideline_path='/home/junze/.jupyter/Train_VAE_full/dataset_1.csv.npy',
                 mergeline_path='/home/junze/.jupyter/Train_VAE_full/dataset_2.csv.npy'):
        self.type = type
        self.action_dim = action_dim

        # Define action and observation spaces for 'continuous' type
        if type == 'L1 movement':
            self.action_space = spaces.Box(low=np.array([-0.5, 0.2]), high=np.array([0.5, 1.0]), dtype=np.float32)
            self.observation_space = spaces.Box(low=np.array([0, -np.pi, -5.0, -5.0, -2.0]),
                                                high=np.array([10000, np.pi, 30.0, 30.0, 2.0]), dtype=np.float32)
            self.init_space = spaces.Box(low=np.array([0, 0.45, 2.0, 1.0, -0.1]),
                                         high=np.array([1, 0.46, 2.2, 1.1, 0.2]))

        elif type == 'L2 movement':
            self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
            self.observation_space = spaces.Box(low=np.array([0, -np.pi, -5.0, -5.0, -2.0]),
                                                high=np.array([10000, np.pi, 30.0, 30.0, 2.0]), dtype=np.float32)
            self.init_space = spaces.Box(low=np.array([0, 0.45, 2.0, 1.0, -0.1]),
                                            high=np.array([1, 0.46, 2.2, 1.1, 0.2]))

        elif type == 'L3 movement':
            self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
            self.observation_space = spaces.Box(low=np.array([0, -np.pi, -5.0, -5.0, -2.0]),
                                                high=np.array([10000, np.pi, 30.0, 30.0, 2.0]), dtype=np.float32)
            self.init_space = spaces.Box(low=np.array([0, 0.45, 2.0, 1.0, -0.1]),
                                            high=np.array([1, 0.46, 2.2, 1.1, 0.2]))
        elif type == 'L4 movement':
            self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
            self.observation_space = spaces.Box(low=np.array([0, -np.pi, -5.0, -5.0, -2.0]),
                                                high=np.array([10000, np.pi, 30.0, 30.0, 2.0]), dtype=np.float32)
            self.init_space = spaces.Box(low=np.array([0, 0.45, 2.0, 1.0, -0.1]),
                                            high=np.array([1, 0.46, 2.2, 1.1, 0.2]))

        # Load and process guideline and mergeline
        self.guideline_raw = self.load_trajectory_data(guideline_path)
        self.mergeline_raw = self.load_trajectory_data(mergeline_path)

        # Convert raw lat/lon data to meters
        self.guideline_meters = self.convert_lat_lon_to_meters(self.guideline_raw)
        self.mergeline_meters = self.convert_lat_lon_to_meters(self.mergeline_raw)

        # Calculate trajectory statistics
        self.guideline_stats = self.calculate_trajectory_statistics(self.guideline_meters)
        self.mergeline_stats = self.calculate_trajectory_statistics(self.mergeline_meters)

        # Store the mean and standard-deviation-based trajectories
        self.mean_trajectory_guideline = self.guideline_stats["avg_trajectory"]
        self.std_trajectory_guideline_lower = self.guideline_stats["lower_trajectory_std"]
        self.std_trajectory_guideline_upper = self.guideline_stats["upper_trajectory_std"]

        self.mean_trajectory_mergeline = self.mergeline_stats["avg_trajectory"]
        self.std_trajectory_mergeline_lower = self.mergeline_stats["lower_trajectory_std"]
        self.std_trajectory_mergeline_upper = self.mergeline_stats["upper_trajectory_std"]

        self.ship_data = None
        self.name_experiment = None
        self.last_pos = np.zeros(3)
        self.last_action = np.zeros(self.action_dim)
        self.simulator = Simulator()
        
        # Set initial guideline and mergeline to their mean trajectories
        self.guideline = self.mean_trajectory_guideline
        self.mergeline = self.mean_trajectory_mergeline
        self.point_coming_ship = self.mergeline[1]
        self.borders = self.calculate_borders(self.guideline, self.mergeline)
        self.viewer = None
        self.test_performance = False
        self.switch = False
        self.init_test_performance = np.linspace(0.45, 0.46, 10)
        self.counter = 0
        self.single_step = 0
        self.start_pos = self.guideline[0]
        self.action_space_other = [0.0, 0.5]
        self.other_ship_state = (0, -2.25, 2.0, 1.0, 0.0, 0.0)
        self.times = np.arange(0, len(self.guideline), 1)
        self.rewardmode = False
        self.attacking_speed = np.abs(self.cal_attacking_speed(self.times, self.single_step, self.mergeline))

    def load_trajectory_data(self, npy_file):
        # Load the guideline or mergeline from a .npy file
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
    
    def enable_evaluation_mode(self):
        self.evaluation_mode = True
        self.eval_data = ShipExperiment(info="Evaluation Run")
        
    def _ensure_experiment_directory(self):
        """Ensure that the '_experiments' directory exists."""
        os.makedirs('_experiments', exist_ok=True)

    def save_experiment(self, descr='_experiment'):
        """
        Use this method save an experiment in .pickle format
        """
        self._ensure_experiment_directory()
        st = datetime.datetime.now().strftime('%Y-%m-%d-%H')
        name = f"{st}{descr}"
        try:
            with open(os.path.join('_experiments', name), 'wb') as f:
                pickle.dump(self.__dict__, f, 2)
            print(f"Experiment saved successfully: {name}")
        except Exception as e:
            print(f"Error saving experiment: {str(e)}")

    def load_from_experiment(self, name):
        """
        Use this method to load an experiment from a .pickle format
        """
        try:
            with open(os.path.join('_experiments', name), 'rb') as f:
                tmp_dict = pickle.load(f)
            self.__dict__.update(tmp_dict)
            print(f"Experiment loaded successfully: {name}")
        except FileNotFoundError:
            print(f"Experiment file not found: {name}")
        except Exception as e:
            print(f"Error loading experiment: {str(e)}")
            
    def convert_lat_lon_to_meters(self, lat_lon_data):
        # Initialize an empty list to store the converted coordinates
        meters_data = []
        R = 6371000  # Earth's radius in meters

        if len(lat_lon_data) == 0:
            return np.array(meters_data)  # Return empty if no data

        # Get the first point to define the origin
        first_lon, first_lat = lat_lon_data[0]
        first_lon_rad = np.radians(first_lon)
        first_lat_rad = np.radians(first_lat)

        origin_x = R * first_lon_rad * np.cos(first_lat_rad)
        origin_y = R * first_lat_rad

        for lon, lat in lat_lon_data:
            lon_rad = np.radians(lon)
            lat_rad = np.radians(lat)

            x = np.abs(R * lon_rad * np.cos(first_lat_rad) - origin_x)
            y = np.abs(R * lat_rad - origin_y)

            meters_data.append([x, y])

        return np.array(meters_data)
    
    def cal_attacking_speed(self, times, single_step, mergeline):
        """
        Calculate the speed at a specific time step using mergeline data.
        Returns:
        float: Speed at the specified time step
        """
        
        print(f"Shape of times: {times.shape}")
        print(f"Shape of mergeline: {mergeline.shape}")
        print(f"single_step: {single_step}")
        
        # Ensure single_step is within bounds (including 0)
        if single_step < 0 or single_step >= len(mergeline):
            raise ValueError(f"single_step ({single_step}) is out of bounds for mergeline (length {len(mergeline)})")
        
        # Get target time from mergeline
        target_time = mergeline[single_step, 0]  # Assuming the first column is time
        print(f"Target time: {target_time}")
        
        # Find the index where time is closest to target_time
        time_diff = np.abs(times - target_time)
        time_index = np.argmin(time_diff)
        
        print(f"Closest time index: {time_index}")
        
        # Ensure we have at least two points to calculate speed
        if time_index == 0:
            start_index, end_index = 0, 1
        elif time_index == len(times) - 1:
            start_index, end_index = -2, -1
        else:
            start_index, end_index = time_index - 1, time_index
        
        # Calculate time difference
        time_diff = times[end_index] - times[start_index]
        
        # Calculate distance difference (assuming second column of mergeline is position)
        dist_diff = mergeline[end_index, 1] - mergeline[start_index, 1]
        
        # Calculate speed
        if time_diff > 0:
            speed = dist_diff / time_diff
        else:
            speed = 0  # Avoid division by zero
        
        return speed

    def calculate_trajectory_statistics(self, data_meters):
        if data_meters.ndim == 2:
            num_points = data_meters.shape[0]
            # Reshape assuming a single trajectory of 'num_points' with 2D (x, y) coordinates
            data_meters = data_meters.reshape(1, num_points, 2)

        # Calculate statistics for each time stamp
        avg_trajectory = np.mean(data_meters, axis=0)
        std_trajectory = np.std(data_meters, axis=0)

        # Percentile-based selection
        lower_bound_percentile = np.percentile(data_meters, 2.5, axis=0)
        upper_bound_percentile = np.percentile(data_meters, 97.5, axis=0)

        # 3-standard-deviation selection
        lower_bound_std = avg_trajectory - 3 * std_trajectory
        upper_bound_std = avg_trajectory + 3 * std_trajectory

        # Find the trajectory closest to the statistical bounds
        def find_closest_trajectory(trajectories, target):
            # Now, 'trajectories' should be 3D (num_trajectories, num_points, 2)
            distances = np.sum(np.sqrt(np.sum((trajectories - target)**2, axis=2)), axis=1)
            return trajectories[np.argmin(distances)]

        # Find closest trajectories to bounds
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
    
    def safe_distance(self, x1, y1, x2, y2):
        if np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) and (self.switch==False):
            self.switch = True
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        
    
    def step(self, action1):
        # According to the action space a different kind of action is selected
        if self.type == 'L1 movement' and self.action_dim == 2:
            angle_action = action1[0]
            rot_action = (action1[1]+1)/10
            angle_action_other = self.action_space_other[0]
            rot_action_other = (self.action_space_other[1]+1)/10             
        state_prime = self.simulator.step(angle_level=angle_action, rot_level=rot_action)
        self.other_ship_state = self.simulator.step(angle_level=angle_action_other, rot_level=rot_action_other)
        self.point_coming_ship = [self.other_ship_state[0], self.other_ship_state[1]]
        obs = self.convert_state(state_prime)
        dn = self.end(state_prime=state_prime, obs=obs)
        rew = self.calculate_reward(obs=obs)
        otherstates = self.point_coming_ship
        self.single_step += 1
        self.last_pos = [state_prime[0], state_prime[1], state_prime[2]]
        self.last_action = np.array([angle_action, rot_action])
        # tcpa, dcpa, cr = self.calculate_safety_method(state_prime)
        print('obs=',obs)
        print('state_prime=',state_prime)
        print('reward=',rew)
        rewardmode = bool(np.sqrt((self.last_pos[0] - self.point_coming_ship[0])**2 + (self.last_pos[1] - self.point_coming_ship[1])**2) - 1500)
        if hasattr(self, 'evaluation_mode') and self.evaluation_mode:
            self.eval_data.new_transition(state_prime, obs, self.last_action, rew, otherstates, rewardmode, self.guideline_meters)
        info = dict()
        return obs, rew, dn, info


    def calculate_safety_method(self, state_prime):
        tcpa = self.tcpa_cal(state_prime[3], state_prime[4], state_prime[0], state_prime[1], self.point_coming_ship[0])
        dcpa = self.dcpa_cal(state_prime[3], state_prime[4], state_prime[0], state_prime[1], self.point_coming_ship[0])
        cr = self.cr_cal(tcpa, dcpa, 2000, np.sqrt(state_prime[3]**2 + state_prime[4]**2), 0.2)
        return tcpa, dcpa, cr
    
    def judging_area(self, x, y):
        i = 0
        if (x > self.guideline[i][0]) and y > (self.guideline[i][1]):
                i += 1
        return i
        
    def calculate_distance_to_guideline(self, x, y):
        ship_point = Point(x, y)
        
        step_index = self.judging_area(x, y)
        
        # Check if we have enough guideline points
        if step_index + 1 >= len(self.guideline):
            print(f"Warning: step_index ({step_index}) is out of bounds for guideline (length {len(self.guideline)})")
            return 0.0
        start_point = Point(self.guideline[step_index])
        end_point = Point(self.guideline[step_index + 1])
        guideline_segment = LineString([start_point, end_point])
        # Calculate the distance
        d = ship_point.distance(guideline_segment)

        return d

    def convert_state(self, state):
        d = self.calculate_distance_to_guideline(state[0], state[1])
        theta = state[2]  # radians
        vx = state[3]  # m/s
        vy = state[4]  # m/s
        thetadot = state[5]  # graus/min
        obs = np.array([d, theta, vx, vy, thetadot])
        return obs

    def calculate_reward(self, obs):
        d, theta, vx, vy, thetadot = obs[0], obs[1]*180/np.pi, obs[2], obs[3], obs[4]*180/np.pi
        if not self.rewardmode:
            return abs((1-(d/1000)))*(1-(d/1000))
        if self.rewardmode:
            return 2+10*(1-np.sqrt((self.last_pos[0]-self.point_coming_ship[0])**2+(self.last_pos[1]-self.point_coming_ship[1])**2)/200)

    def set_init_space(self, low, high):
        self.init_space = spaces.Box(low=np.array(low), high=np.array(high))

    def end(self, state_prime, obs):
        if (self.switch) and self.safe_distance(state_prime[0], state_prime[1], self.point_coming_ship[0], self.point_coming_ship[1]) < 300:
                return True
        elif not self.observation_space.contains(obs):
                print("\n Smashed")
                print("steps: ", self.single_step)
                return True
        return False



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
            init_states = np.array([
                self.start_pos[0], self.start_pos[1],  # x, y
                angle,  # theta
                v_lon * np.cos(angle), v_lon * np.sin(angle),  # vx, vy
                0  # thetadot
            ], dtype=float)
            self.counter += 1
        else:
            init_states = np.array([
                self.start_pos[0], self.start_pos[1],  
                init[1],  
                init[2] * np.cos(init[1]), init[2] * np.sin(init[1]),  
                0  
            ], dtype=float)
        self.single_step = 0
        self.simulator.reset_start_pos(init_states)
        self.last_pos = np.array([init_states[0], init_states[1], init_states[2]])
        self.point_coming_ship = self.mergeline[0]
        state = self.simulator.get_state()
        self.single_step = 0
        print('Resetting position')
        otherstates = self.point_coming_ship
        rewardmode = 0
        if hasattr(self, 'evaluation_mode') and self.evaluation_mode:
            if self.eval_data.iterations > 0:
                self.eval_data.save_experiment(f"eval_run_{self.eval_data.iterations}")
            self.eval_data.new_iter(state, self.convert_state(state), np.zeros(self.action_dim), np.array([0]), otherstates, rewardmode, self.guideline_meters)

        if self.viewer is not None:
            self.viewer.end_episode()

        return self.convert_state(state)
    
    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = Viewer()
            self.viewer.plot_boundary(self.borders)
            self.viewer.plot_guidance_line(self.point_a, self.point_b)

        img_x_pos = self.last_pos[0] - self.point_b[0] * (self.last_pos[0] // self.point_b[0])
        if self.last_pos[0]//self.point_b[0] > self.number_loop:
            self.viewer.end_episode()
            self.viewer.plot_position(img_x_pos, self.last_pos[1], self.last_pos[2], 20 * self.last_action[0])
            self.viewer.restart_plot()
            self.number_loop += 1
        else:
            self.viewer.plot_position(img_x_pos, self.last_pos[1], self.last_pos[2], 20 * self.last_action[0])

    def close(self, ):
        self.viewer.freeze_scream()

    def set_save_experice(self, name='experiment_ssn_ppo_10iter'):
        assert type(name) == type(""), 'name must be a string'
        self.ship_data = ShipExperiment()
        self.name_experiment = name

    def set_test_performance(self):
        self.test_performance = True
    
    def save_evaluation_data(self):
        if hasattr(self, 'evaluation_mode') and self.evaluation_mode:
            self.eval_data.save_experiment("final_evaluation_data")


if __name__ == '__main__':
    mode = 'test'  
    if mode == 'train':
        env = ShipEnv()
        ShipExp = ShipExperiment()
        for i_episode in range(10):
            obs = env.reset()
            for t in range(10000):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break
        env.close()
    elif mode == 'test':
        env = ShipEnv()
        #testing samples
        Simulator = Simulator()
        Simulator.test_straight_line_motion()
        Simulator.test_various_scenarios()
        Simulator.test_turning_scenario(initial_state=[0, 0, 0, 5, 0, 0], rudder_angle=0.1, propulsion=1)