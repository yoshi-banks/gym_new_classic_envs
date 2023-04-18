import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import gym_new_classic_envs.envs.mass.mass_resources.massParam as P
from gym_new_classic_envs.envs.mass.mass_resources.massDynamics import massDynamics
from gym_new_classic_envs.envs.mass.mass_controllers.PID.massController import massController
from gym_new_classic_envs.utils.signalGenerator import signalGenerator
from gym_new_classic_envs.envs.mass.mass_resources.massAnimation import massAnimation
from gym_new_classic_envs.envs.mass.mass_resources.massDataPlotter import dataPlotter
from gym_new_classic_envs.envs.mass.mass_env import MassEnv


env = gym.make('Mass-v0', render_mode='human')

done = False
i = 0
env.reset()
while not done:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if i > 100:
        env.render()
        i = 0
    i += 1
    print(observation, reward, terminated, truncated, info)

# Keeps the program from closing until the user presses a button
print('Press key to close')
plt.waitforbuttonpress()
plt.close()