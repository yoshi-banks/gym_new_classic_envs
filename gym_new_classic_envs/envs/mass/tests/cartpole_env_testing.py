#%% Imports
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import gym_new_classic_envs.envs.mass.mass_resources.massParam as P
from gym_new_classic_envs.envs.mass.mass_resources.massDynamics import massDynamics
from gym_new_classic_envs.envs.mass.mass_controllers.PID.massController import massController
from gym_new_classic_envs.utils.signalGenerator import signalGenerator
from gym_new_classic_envs.envs.mass.mass_resources.massAnimation import massAnimation
from gym_new_classic_envs.envs.mass.mass_resources.massDataPlotter import dataPlotter
from gym_new_classic_envs.envs.mass.mass_env import MassEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

#%% Create env

# env = gym.make('Mass-v0')
env = gym.make('CartPole-v1')

model = PPO('MlpPolicy', env, verbose=1)

#%% Evaluate model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
# %%
