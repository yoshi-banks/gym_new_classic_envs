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

def evaluate(model, num_episodes=100):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
#     env = model.get_env()
#     all_episode_rewards = []
#     for i in range(num_episodes):
#         print('episode:', i)
#         episode_rewards = []
#         done = False
#         obs = env.reset()
#         while not done:
#             # _states are only useful when using LSTM policies
#             action, _states = model.predict(obs)
#             # here, action, rewards and dones are arrays
#             # because we are using vectorized env
#             obs, reward, terminated, info = env.step(action)
#             done = terminated
#             episode_rewards.append(reward)

#         all_episode_rewards.append(sum(episode_rewards))

#     mean_episode_reward = np.mean(all_episode_rewards)
#     print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

#     return mean_episode_reward

# #%% Create env

# env = gym.make('Mass-v0')
# # env = gym.make('CartPole-v1')

# model = PPO('MlpPolicy', env, verbose=1)

# #%%
# evaluate(model, num_episodes=100)

# #%%
# env.reset()
# u = np.array([0])
# print(u.shape)
# action = env.action_space.sample()
# print(action)
# action = np.array([2.0]).reshape(action.shape)
# print(action)
# nmt = 6000
# t = np.linspace(1,nmt,nmt)
# state = np.ones((nmt,2))
# for i in range(nmt):
#     # print('i:', i, 'state:', env.state)
#     state[i,0] = env.state[0]
#     state[i,1] = env.state[1]
#     obs, reward, terminated, truncated, _ = env.step(action)
#     if terminated or truncated:
#         print('terminated:', terminated, 'truncated:', truncated)
#         break

# plt.figure(1)
# plt.plot(t,state[:,0], label=r'$\theta$')
# plt.plot(t,state[:,1], label=r'$\omega$')
# plt.legend()

# plt.figure(2)
# plt.plot(t,np.ones(nmt)*action, label=r'$u$')
# plt.legend()

# #%%
# evaluate(model, num_episodes=100)

# #%% Evaluate model
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
# # %%

# model.learn(total_timesteps=1000000)

# #%% Evaluate model
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
# # %% Save model

# # Create save dir
# # save_dir = "/tmp/gym/"
# save_dir = "C:\\tmp\\gym\\"
# os.makedirs(save_dir, exist_ok=True)
# model.save(save_dir + "ppo_mass_testing2")

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

