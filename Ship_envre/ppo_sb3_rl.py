import gym
import numpy as np
import tensorboard
import tensorflow as tf

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure

from Ship_env import ShipEnv
from ship_data import ShipExperiment
from viewer import Viewer

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

shipExp = ShipExperiment()

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=500000)

PPO.save(path="/home/junze/.jupyter/Digitaltesting/Evaluation/PPO")


mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

if __name__ == '__main__':
    mode = 'train'
    if mode == 'train':
        env = ShipEnv()
        for i_episode in range(10):
            obs = env.reset()
            for t in range(10000):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break
        env.close()
    elif mode == 'eval':
        env.set_test_performance()
        env.set_save_experice()





