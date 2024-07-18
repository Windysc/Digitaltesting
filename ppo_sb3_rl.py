import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure

from ship_env import ShipEnv
from ship_data import ShipExperiment

# Create the environment
env = ShipEnv()
# Introduce the vectorized environment

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=100000)

model = PPO.load("ppo_ship")

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

if __name__ == '__main__':
    mode = 'train'
    if mode == 'train':
        env = ShipEnv()
        shipExp = ShipExperiment()
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





