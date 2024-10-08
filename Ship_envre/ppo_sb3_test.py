import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
from Ship_env import ShipEnv

# Assuming ShipEnv is already registered with gym
# from Ship_env import ShipEnv

# Create a single environment and wrap it with a Monitor
env = ShipEnv()
env = Monitor(env)

# Wrap the environment in a DummyVecEnv for compatibility with Stable Baselines 3
env = DummyVecEnv([lambda: env])

# Initialize PPO model
model = PPO("MlpPolicy", env, verbose=1)


total_timesteps = 1_000_000  
update_timestep = 2048  
n_epochs = 10  
batch_size = 64  


episode_rewards = []
episode_lengths = []


obs = env.reset()
episode_reward = 0
episode_length = 0
global_step = 0

while global_step < total_timesteps:
    done = False
    episode_reward = 0
    episode_length = 0

    while not done:
        # Choose action using the current policy
        action, _states = model.predict(obs)
        
        obs, reward, done, info = env.step(action)

        episode_reward += reward
        episode_length += 1
        global_step += 1


    # At the end of an episode, reset the environment
    print(f"Episode finished: Reward = {episode_reward}, Length = {episode_length}")
    episode_rewards.append(episode_reward)
    episode_lengths.append(episode_length)
    
    # Reset environment and continue
    obs = env.reset()

# After training, plot the rewards
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("PPO Training Progress")
plt.show()
