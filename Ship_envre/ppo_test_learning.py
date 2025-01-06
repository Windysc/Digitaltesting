import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os
from datetime import datetime
import json
import matplotlib.pyplot as plt
from Ship_env import ShipEnv

def ensure_numpy_array(data):
    """Convert list or array-like object to numpy array safely"""
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, list):
        return np.array(data)
    else:
        return np.array([data])

def safe_close_env(env):
    """Safely close the environment"""
    if env is not None:
        try:
            if hasattr(env, 'close'):
                env.close()
            if hasattr(env, 'viewer') and env.viewer is not None:
                env.viewer = None
        except Exception as e:
            print(f"Warning: Non-critical error while closing environment: {str(e)}")

def create_episode_plots(episode_data, episode_num, save_dir):
    """Create plots for a single evaluation episode with error handling"""
    try:
        # Ensure data is in numpy array format
        positions = ensure_numpy_array(episode_data['positions'])
        target_positions = ensure_numpy_array(episode_data['target_positions'])
        actions = ensure_numpy_array(episode_data['actions'])
        rewards = ensure_numpy_array(episode_data['rewards'])
        
        # Trajectory plot
        plt.figure(figsize=(10, 10))
        plt.plot(positions[:, 0], positions[:, 1], 'b-', label='Ship')
        plt.plot(target_positions[:, 0], target_positions[:, 1], 'r-', label='Target')
        plt.title(f'Episode {episode_num + 1} - Ship Trajectory')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.grid(True)
        plt.legend()
        
        plt.savefig(os.path.join(save_dir, f'trajectory_episode_{episode_num+1}.png'))
        plt.close()
        
        # Actions and rewards plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        time_steps = np.arange(len(actions))
        ax1.plot(time_steps, actions)
        ax1.set_title('Actions Over Time')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Action Values')
        ax1.grid(True)
        ax1.legend(['Angle', 'Rotation'])
        
        ax2.plot(time_steps, rewards, 'g-')
        ax2.set_title('Rewards Over Time')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Reward')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'actions_rewards_episode_{episode_num+1}.png'))
        plt.close()
        
    except Exception as e:
        print(f"Warning: Error creating plots for episode {episode_num + 1}: {str(e)}")

def run_evaluation(env, model, num_episodes=10):
    env.enable_evaluation_mode()
    """Run evaluation episodes with improved error handling"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"evaluation_results_{timestamp}"
    data_dir = os.path.join(base_dir, "data")
    plot_dir = os.path.join(base_dir, "plots")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    evaluation_results = {'episodes': []}
    
    for episode in range(num_episodes):
        print(f"\nRunning episode {episode + 1}/{num_episodes}")
        episode_data = {
            'positions': [],
            'target_positions': [],
            'actions': [],
            'rewards': [],
            'observations': [],
            'total_reward': 0,
            'steps': 0
        }
        
        obs = env.reset()
        done = False
        
        while not done:
            try:
                # Store pre-step data
                if hasattr(env, 'last_pos'):
                    current_pos = env.last_pos
                    if isinstance(current_pos, np.ndarray):
                        current_pos = current_pos.tolist()
                    elif not isinstance(current_pos, list):
                        current_pos = [float(current_pos)]
                    episode_data['positions'].append(current_pos)
                
                if hasattr(env, 'point_coming_ship'):
                    target_pos = env.point_coming_ship
                    if isinstance(target_pos, np.ndarray):
                        target_pos = target_pos.tolist()
                    elif not isinstance(target_pos, list):
                        target_pos = [float(target_pos)]
                    episode_data['target_positions'].append(target_pos)
                
                # Get action from model
                action, _ = model.predict(obs, deterministic=True)
                
                # Take step in environment
                obs, reward, done, info = env.step(action)
                
                # Store post-step data
                if isinstance(action, np.ndarray):
                    action = action.tolist()
                episode_data['actions'].append(action)
                episode_data['rewards'].append(float(reward))
                episode_data['observations'].append(obs.tolist() if isinstance(obs, np.ndarray) else obs)
                episode_data['total_reward'] += reward
                episode_data['steps'] += 1
                
            except Exception as e:
                print(f"Warning: Error during episode step: {str(e)}")
                done = True
        
        print(f"Episode {episode + 1} completed: {episode_data['steps']} steps, "
              f"reward: {episode_data['total_reward']:.2f}")
        
        evaluation_results['episodes'].append(episode_data)
        create_episode_plots(episode_data, episode, plot_dir)
    
    # Save results
    try:
        with open(os.path.join(data_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(evaluation_results, f)
    except Exception as e:
        print(f"Warning: Error saving evaluation results: {str(e)}")
    
    return evaluation_results

def main():
    env = None
    try:
        print("Creating environment...")
        env = ShipEnv()
        
        print("Loading model...")
        model = PPO.load("ppo_ship ship12")
        model.set_env(env)
        
        # Quick evaluation
        mean_reward, std_reward = evaluate_policy(
            model, 
            env,
            n_eval_episodes=10,
            deterministic=True
        )
        print(f"\nQuick evaluation results:")
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        
        # Detailed evaluation
        evaluation_results = run_evaluation(env, model, num_episodes=10)
        print("\nDetailed evaluation completed successfully")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise
    finally:
        safe_close_env(env)

if __name__ == '__main__':
    main()