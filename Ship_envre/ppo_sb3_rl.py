import gym
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation, PillowWriter
import json
from datetime import datetime

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

# def run_evaluation_and_visualize(env, model, num_episodes=10):
#     env.enable_evaluation_mode()
    
#     for episode in range(num_episodes):
#         obs = env.reset()
#         done = False
#         total_reward = 0
#         steps = 0
        
#         while not done:
#             action, _states = model.predict(obs, deterministic=True)
#             obs, reward, done, info = env.step(action)
#             total_reward += reward
#             steps += 1
        
#         print(f"Episode {episode+1} completed. Steps: {steps}, Total Reward: {total_reward}")
    
#     env.save_evaluation_data()
    
#     # Load the saved data
#     eval_data = ShipExperiment()
#     try:
#         eval_data.load_from_experiment("/home/junze/_experiments/2024-10-11-12final_evaluation_data")
#     except FileNotFoundError:
#         print("Evaluation data file not found. Make sure the file was saved correctly.")
#         return
    
#     # Create a directory to save plots
#     plot_dir = "evaluation_plots"
#     os.makedirs(plot_dir, exist_ok=True)
    
#     # Visualize the data
#     try:
#         for i in range(eval_data.iterations + 1):
#             visualize_episode_data(eval_data, i, plot_dir)
#     except Exception as e:
#         print(f"Error during visualization: {str(e)}")

# def visualize_episode_data(eval_data, episode, plot_dir):
#     """Visualize data for a single episode, saving plots as image files."""
#     try:
#         # Plot trajectory
#         states = np.array(eval_data.states[episode])
#         other_states = np.array(eval_data.otherstates[episode])
        
#         if len(states) != len(other_states):
#             print(f"Warning: Mismatch in data lengths for episode {episode}. Trimming to shorter length.")
#             min_len = min(len(states), len(other_states))
#             states = states[:min_len]
#             other_states = other_states[:min_len]
        
#         plt.figure(figsize=(10, 8))
#         plt.plot(states[:, 0], states[:, 1], label='Vessel')
#         plt.plot(other_states[:, 0], other_states[:, 1], label='Objective Vessel')
#         plt.legend()
#         plt.title(f'Trajectory - Episode {episode}')
#         plt.xlabel('X position')
#         plt.ylabel('Y position')
#         plt.savefig(os.path.join(plot_dir, f'trajectory_episode_{episode}.png'))
#         plt.close()
        
#         # Plot actions
#         actions = np.array(eval_data.actions[episode])
#         plt.figure(figsize=(10, 6))
#         for i in range(actions.shape[1]):
#             plt.plot(actions[:, i], label=f'Action {i+1}')
#         plt.legend()
#         plt.title(f'Actions - Episode {episode}')
#         plt.xlabel('Step')
#         plt.ylabel('Action Value')
#         plt.savefig(os.path.join(plot_dir, f'actions_episode_{episode}.png'))
#         plt.close()
        
#         # Plot reward
#         rewards = np.array(eval_data.rewards[episode])
#         plt.figure(figsize=(10, 6))
#         plt.plot(rewards)
#         plt.title(f'Reward - Episode {episode}')
#         plt.xlabel('Step')
#         plt.ylabel('Reward')
#         plt.savefig(os.path.join(plot_dir, f'reward_episode_{episode}.png'))
#         plt.close()
        
#         # Plot observations
#         observations = np.array(eval_data.observations[episode])
#         plt.figure(figsize=(12, 8))
#         for i in range(observations.shape[1]):
#             plt.plot(observations[:, i], label=f'Obs {i+1}')
#         plt.legend()
#         plt.title(f'Observations - Episode {episode}')
#         plt.xlabel('Step')
#         plt.ylabel('Observation Value')
#         plt.savefig(os.path.join(plot_dir, f'observations_episode_{episode}.png'))
#         plt.close()
        
#         print(f"Plots for episode {episode} saved in {plot_dir}")
        
#     except Exception as e:
#         print(f"Error visualizing data for episode {episode}: {str(e)}")
def run_evaluation_and_visualize(env, model, num_episodes=10):
    """
    Run evaluation episodes and save both numerical results and animations
    """
    env.enable_evaluation_mode()
    
    # Create directories for saving results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"evaluation_results_{timestamp}"
    gif_dir = os.path.join(base_dir, "animations")
    data_dir = os.path.join(base_dir, "data")
    plot_dir = os.path.join(base_dir, "plots")
    
    for directory in [gif_dir, data_dir, plot_dir]:
        os.makedirs(directory, exist_ok=True)
    
    evaluation_results = {
        'episodes': []
    }
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_data = {
            'last_pos': [],
            'other_states': [],
            'actions': [],
            'rewards': [],
            'observations': [],
            'total_reward': 0,
            'steps': 0
        }
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # Store step data
            episode_data['last_pos'].append(env.last_pos)
            episode_data['other_states'].append(env.point_coming_ship)
            episode_data['actions'].append(action.tolist())
            episode_data['rewards'].append(float(reward))
            episode_data['observations'].append(obs.tolist())
            episode_data['total_reward'] += reward
            episode_data['steps'] += 1
        
        # Save episode data
        evaluation_results['episodes'].append(episode_data)
        
        # Create and save static plots
        create_episode_plots(episode_data, episode, plot_dir)
        
        print(f"Episode {episode+1} completed. Steps: {episode_data['steps']}, "
              f"Total Reward: {episode_data['total_reward']:.2f}")
    
    # Save all numerical results
    with open(os.path.join(data_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(evaluation_results, f)
    
    # Create summary plots
    create_summary_plots(evaluation_results, plot_dir)

def create_episode_plots(episode_data, episode_num, save_dir):
    """
    Create static plots for the episode
    """
    # Trajectory plot
    plt.figure(figsize=(10, 10))
    last_pos = np.array(episode_data['last_pos'])
    other_states = np.array(episode_data['other_states'])
    plt.plot(last_pos[:, 0], last_pos[:, 1], 'b-', label='Main Ship')
    plt.plot(other_states[:, 0], other_states[:, 1], 'r-', label='Target Ship')
    plt.title(f'Episode {episode_num + 1} - Complete Trajectory')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'trajectory_episode_{episode_num+1}.png'))
    plt.close()
    
    # Actions and rewards plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    actions = np.array(episode_data['actions'])
    ax1.plot(actions)
    ax1.set_title('Actions')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Action Value')
    ax1.grid(True)
    
    rewards = np.array(episode_data['rewards'])
    ax2.plot(rewards)
    ax2.set_title('Rewards')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Reward')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'actions_rewards_episode_{episode_num+1}.png'))
    plt.close()

def create_summary_plots(evaluation_results, save_dir):
    """
    Create summary plots across all episodes
    """
    episodes = evaluation_results['episodes']
    num_episodes = len(episodes)
    
    # Episode statistics
    total_rewards = [ep['total_reward'] for ep in episodes]
    episode_lengths = [ep['steps'] for ep in episodes]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    ax1.plot(range(1, num_episodes + 1), total_rewards, 'bo-')
    ax1.set_title('Total Rewards per Episode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)
    
    ax2.plot(range(1, num_episodes + 1), episode_lengths, 'ro-')
    ax2.set_title('Episode Lengths')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Number of Steps')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'evaluation_summary.png'))
    plt.close()


class VisualShipEnv(ShipEnv):
    def __init__(self):
        super().__init__()
        self.viewer = Viewer()
        
    def render(self):
        x, y, theta, v, omega = self.state
        rudder_angle = self.get_rudder_angle()  
        self.viewer.plot_position(x, y, theta, rudder_angle)

# Create the environment
env = ShipEnv()
# Introduce the vectorized environment
print(env.borders)

shipExp = ShipExperiment()

model = PPO("MlpPolicy", env, verbose=1)

print("Model's action space:", model.action_space)

print("Environment's action space:", env.action_space)


if __name__ == '__main__':
    mode = 'eval'
    if mode == 'train':
            obs = env.reset()
            action1, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action1)
            model.learn(total_timesteps=int(4e5))
            model.save("ppo_ship ship13")
            print('Training Done')
    elif mode == 'eval':
        del model
        model = PPO.load("ppo_ship ship12")
        env.set_test_performance()
        env.set_save_experice()
        run_evaluation_and_visualize(env, model)



