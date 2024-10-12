import gym
import numpy as np
import tensorboard
import tensorflow as tf
import tqdm
import matplotlib.pyplot as plt
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure

from Ship_env import ShipEnv
from ship_data_vis import ShipExperiment
from viewer import Viewer

class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        if self.locals.get("dones"):
            self.episode_rewards.append(sum(self.locals["rewards"]))
            self.episode_lengths.append(self.n_calls - sum(self.episode_lengths))
        return True

    def _on_training_end(self) -> None:
        print(f"Average episode reward: {sum(self.episode_rewards) / len(self.episode_rewards)}")
        print(f"Average episode length: {sum(self.episode_lengths) / len(self.episode_lengths)}")

# Use the callback during training
callback = RewardLoggingCallback()

def run_evaluation_and_visualize(env, model, num_episodes=10):
    env.enable_evaluation_mode()
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
        
        print(f"Episode {episode+1} completed. Steps: {steps}, Total Reward: {total_reward}")
    
    env.save_evaluation_data()
    
    # Load the saved data
    eval_data = ShipExperiment()
    try:
        eval_data.load_from_experiment("/home/junze/_experiments/2024-10-11-12final_evaluation_data")
    except FileNotFoundError:
        print("Evaluation data file not found. Make sure the file was saved correctly.")
        return
    
    # Create a directory to save plots
    plot_dir = "evaluation_plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Visualize the data
    try:
        for i in range(eval_data.iterations + 1):
            visualize_episode_data(eval_data, i, plot_dir)
    except Exception as e:
        print(f"Error during visualization: {str(e)}")

def visualize_episode_data(eval_data, episode, plot_dir):
    """Visualize data for a single episode, saving plots as image files."""
    try:
        # Plot trajectory
        states = np.array(eval_data.states[episode])
        other_states = np.array(eval_data.otherstates[episode])
        
        if len(states) != len(other_states):
            print(f"Warning: Mismatch in data lengths for episode {episode}. Trimming to shorter length.")
            min_len = min(len(states), len(other_states))
            states = states[:min_len]
            other_states = other_states[:min_len]
        
        plt.figure(figsize=(10, 8))
        plt.plot(states[:, 0], states[:, 1], label='Vessel')
        plt.plot(other_states[:, 0], other_states[:, 1], label='Objective Vessel')
        plt.legend()
        plt.title(f'Trajectory - Episode {episode}')
        plt.xlabel('X position')
        plt.ylabel('Y position')
        plt.savefig(os.path.join(plot_dir, f'trajectory_episode_{episode}.png'))
        plt.close()
        
        # Plot actions
        actions = np.array(eval_data.actions[episode])
        plt.figure(figsize=(10, 6))
        for i in range(actions.shape[1]):
            plt.plot(actions[:, i], label=f'Action {i+1}')
        plt.legend()
        plt.title(f'Actions - Episode {episode}')
        plt.xlabel('Step')
        plt.ylabel('Action Value')
        plt.savefig(os.path.join(plot_dir, f'actions_episode_{episode}.png'))
        plt.close()
        
        # Plot reward
        rewards = np.array(eval_data.rewards[episode])
        plt.figure(figsize=(10, 6))
        plt.plot(rewards)
        plt.title(f'Reward - Episode {episode}')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.savefig(os.path.join(plot_dir, f'reward_episode_{episode}.png'))
        plt.close()
        
        # Plot observations
        observations = np.array(eval_data.observations[episode])
        plt.figure(figsize=(12, 8))
        for i in range(observations.shape[1]):
            plt.plot(observations[:, i], label=f'Obs {i+1}')
        plt.legend()
        plt.title(f'Observations - Episode {episode}')
        plt.xlabel('Step')
        plt.ylabel('Observation Value')
        plt.savefig(os.path.join(plot_dir, f'observations_episode_{episode}.png'))
        plt.close()
        
        print(f"Plots for episode {episode} saved in {plot_dir}")
        
    except Exception as e:
        print(f"Error visualizing data for episode {episode}: {str(e)}")

# class VisualShipEnv(ShipEnv):
#     def __init__(self):
#         super().__init__()
#         self.viewer = Viewer()
        
#     def render(self):
#         # Assuming self.state contains [x, y, theta, v, omega]
#         x, y, theta, v, omega = self.state
#         # Assuming you have a way to get the rudder angle
#         rudder_angle = self.get_rudder_angle()  # You need to implement this method
#         self.viewer.plot_position(x, y, theta, rudder_angle)

# Create the environment
env = ShipEnv()
# Introduce the vectorized environment
print(env.borders)

shipExp = ShipExperiment()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_ship_tensorboard/")

print("Model's action space:", model.action_space)

print("Environment's action space:", env.action_space)


if __name__ == '__main__':
    mode = 'train'
    if mode == 'train':
            obs = env.reset()
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            model.learn(total_timesteps=int(1e5), progress_bar=True, callback=callback)
            model.save("ppo_ship ship9")
            print('Training Done')
    elif mode == 'eval':
        env.set_test_performance()
        env.set_save_experice()
        model = PPO.load("ppo_ship ship8", env=env)
        run_evaluation_and_visualize(env, model)



