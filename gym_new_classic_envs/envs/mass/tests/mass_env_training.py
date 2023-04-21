#%% Imports
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os
import gym_new_classic_envs.envs.mass.mass_resources.massParam as P
from gym_new_classic_envs.envs.mass.mass_resources.massDynamics import massDynamics
from gym_new_classic_envs.envs.mass.mass_controllers.PID.massController import massController
from gym_new_classic_envs.utils.signalGenerator import signalGenerator
from gym_new_classic_envs.envs.mass.mass_resources.massAnimation import massAnimation
from gym_new_classic_envs.envs.mass.mass_resources.massDataPlotter import dataPlotter
from gym_new_classic_envs.envs.mass.mass_env import MassEnv
from stable_baselines3.common.evaluation import evaluate_policy
from gym_new_classic_envs.utils.visualize import record_matplotlib_video
from stable_baselines3 import PPO

#%% Create env
env = gym.make('Mass-v0')
# env = gym.make('CartPole-v1')
model = PPO('MlpPolicy', env, verbose=1)

#%% Load model
env = gym.make('Mass-v0')
save_dir = "C:\\tmp\\gym\\"
model = PPO.load(save_dir + "ppo_mass_testing2", env=env)

#%% Evaluate model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
# %% Train model

model.learn(total_timesteps=1000000)

#%% Evaluate model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
# %% Save model

# Create save dir
# save_dir = "/tmp/gym/"
save_dir = "C:\\tmp\\gym\\"
os.makedirs(save_dir, exist_ok=True)
model.save(save_dir + "ppo_mass_testing-2_1")

# %% Save a video
save_dir = "C:\\tmp\\gym\\"
model = PPO.load(save_dir + "ppo_mass_testing2")

env = gym.make('Mass-v0', render_mode='human')
obs = env.reset()[0]
for i in range(10000):
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = env.step(action)
    env.render()
input('Press enter to close')
env.close()
# record_matplotlib_video(env_id, model, target=target, video_length=1000, prefix='ppo-arm', video_folder='videos/')

