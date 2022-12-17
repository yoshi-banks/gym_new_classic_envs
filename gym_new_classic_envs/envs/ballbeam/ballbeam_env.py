import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

# import os
# import sys
# sys.path.append(os.path.join(sys.path[0], 'custom_arm'))  # add directory

# print(sys.path)

import matplotlib.pyplot as plt

from gym_new_classic_envs.envs.arm.arm_resources.armDynamics import armDynamics
from gym_new_classic_envs.utils.signalGenerator import signalGenerator
import gym_new_classic_envs.envs.arm.arm_resources.armParam as P

RUN_CUSTOM_DYN = True
WATCH_THETA = False

class BallbeamEnv(gym.Env):
    metadata = {'render.modes': ['human'], "video.frames_per_second": 30}

    def __init__(self, target=0.0):
        # Set the target
        self.target = target

        # Initialize state
        self.state = np.array([P.theta0, P.thetadot0])

        # Arm Dynamics is uploading dynamic parameters from armParam.py
        self.arm = armDynamics()
        self.reference = signalGenerator(amplitude=0.0, frequency=0.1)

        # Intialize parameters
        self.t = P.t_start  # time starts at t_start # TODO this may mess things up
        self.max_torque = P.tau_max
        self.max_speed = P.thetadot_max
        self.dt = P.Ts

        # Initialize action space
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )

        # Initialize observation space
        # Watch theta and thetadot
        # TODO tie these into the params.py file
        if WATCH_THETA is True:
            theta_max = 2*np.pi
            theta_min = 0

            high_observation = np.array([theta_max, -self.max_speed], dtype=np.float32)
            low_observation = np.array([theta_min, self.max_speed], dtype=np.float32)
            self.observation_space = spaces.Box(
                low=low_observation, high=high_observation, dtype=np.float32
            )
        else:
            # or watch the position of the end of the pendulum
            high = np.array([1, 1, self.max_speed], dtype=np.float32)
            self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Initialize visualization tools
        self.animation = None
        self.dataPlot = None

        # Initialize seed
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        self.ref = self.reference.step(self.t)
        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u
        
        # Work the magic by manipulating the cost function
        theta = self.arm.state.item(0)
        thetadot = self.arm.state.item(1)
        # costs = (theta - self.target)**2 + 0.1 * thetadot ** 2 + 0.001 * (u ** 2)
        costs = (angle_normalize(theta) - self.target) ** 2 + 0.1 * thetadot ** 2 + 0.001 * (u ** 2)
        # costs = (angle_normalize(theta) - self.target) ** 2
        # costs = (theta - np.pi/4)**2

        # Propagate dynamics in between plot samples
        self._run_dynamics(u)

        # Increment time
        self.t = self.t + self.dt

        return self._get_obs(), -costs, False, {}

    def reset(self):
        # reset dynamic state
        high = np.array([2*np.pi, 1])
        low = np.array([0, -1])
        self.state = self.np_random.uniform(low=low, high=high)
        if RUN_CUSTOM_DYN is True:
            self.arm.reset(self.state[0], self.state[1])

        self.last_u = None
        self.t = P.t_start
        return self._get_obs()

    def _get_obs(self):
        if WATCH_THETA is True:
            return np.array([
                self.state.item(0),
                self.state.item(1)
            ], dtype=np.float32)
        else:
            theta, thetadot = self.state
            return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    def _run_dynamics(self, u):
        if RUN_CUSTOM_DYN is False:
            # print('Running pendulum dynamics')
            # Dynamics from pendulum env
            g = 10.0
            l = 1.0
            m = 1.0
            
            th, thdot = self.state
            newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l ** 2) * u) * self.dt
            newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
            newth = th + newthdot * self.dt
            self._set_state(newth, newthdot)

        if RUN_CUSTOM_DYN is True:
            # print('Running custom dynamics')
            # Dynamics from custom simulator
            y = self.arm.update(u)
            self._set_state(self.arm.state.item(0), self.arm.state.item(1))

    def _set_state(self, theta, thetadot):
        self.state = np.array([theta, thetadot])

    def render(self, mode='human'):
        if self.animation is None:
            from gym_new_classic_envs.envs.arm.arm_resources.armAnimation import armAnimation

            self.animation = armAnimation()
        else:
            # HACK
            # self.state = np.array([np.pi, 0.0])
            self.animation.update(self.state)

            # the pause causes the figure to be displayed during the
            # simulation
            plt.pause(0.0001)

        if self.dataPlot is None:
            from gym_new_classic_envs.envs.arm.arm_resources.armDataPlotter import dataPlotter
            self.dataPlot = dataPlotter()
        else:
            # print(self.state.item(0))
            self.dataPlot.update(self.t, self.target, self.state, self.last_u)
            # self.dataPlot.update(self.t, self.ref, self.arm.state, self.last_u)

            # the pause causes the figure to be displayed during the
            # simulation
            plt.pause(0.0001)

    def close(self):
        if self.animation:
            self.animation = None
        if self.dataPlot:
            self.dataPlot = None

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi