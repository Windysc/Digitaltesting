import gym
import numpy as np
import tensorboard
import tensorflow as tf
import tqdm
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure

from Ship_env import ShipEnv
from ship_data import ShipExperiment
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

model.learn(total_timesteps=int(5e5), progress_bar=True, callback=callback)

model.save("ppo_ship ship2")

del model  # delete trained model to demonstrate loading

model = PPO.load("ppo_ship ship1", env=env)    

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


if __name__ == '__main__':
    mode = 'train'
    if mode == 'train':
        for i in range(1000):
            obs = env.reset()
            if not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
        print('Training Done')
    elif mode == 'eval':
        env.set_test_performance()
        env.set_save_experice()



