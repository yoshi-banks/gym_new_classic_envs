import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os
import time
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

def evaluate(model, num_episodes=100):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """

save_dir = "C:\\tmp\\gym\\"
model = PPO.load(save_dir + "ppo_mass_testing-2_1")

env = gym.make('Mass-v0', render_mode='human')
obs = env.reset()[0]

time_in_control_loop = 0
for i in range(50000):
    start_time = time.time()
    action, _states = model.predict(obs)
    end_time = time.time()
    time_in_control_loop += end_time - start_time
    obs, rewards, terminated, truncated, info = env.step(action)
    env.render()
    
print('time in control loop: ', time_in_control_loop)
input('Press enter to close')
env.close()
# record_matplotlib_video(env_id, model, target=target, video_length=1000, prefix='ppo-arm', video_folder='videos/')

