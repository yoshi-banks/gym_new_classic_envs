import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from gymnasium.envs.classic_control import utils
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union

from gym_new_classic_envs.envs.mass.mass_resources.massDynamics import massDynamics
from gym_new_classic_envs.utils.signalGenerator import signalGenerator
from gym_new_classic_envs.envs.mass.mass_resources.massDataPlotter import dataPlotter
from gym_new_classic_envs.envs.mass.mass_resources.massAnimation import massAnimation
import gym_new_classic_envs.envs.mass.mass_resources.massParam as P
from gym_new_classic_envs.envs.mass.mass_controllers.PID.massController import massController

class MassEnv(gym.Env):
    metadata = {
        'render_modes': ['human'],
        'render_fps': 30,
    }

    def __init__(self, render_mode=None):
        # print('Initializing Mass Env...')

        self.render_mode = render_mode
        self.ref = 0.0
        self.count = 0
        self.max_count = 5000
        self.render_count = 0
        self.render_max_count = 100
        
        # instantiate mass, controller, and reference classes
        self.mass = massDynamics(alpha=0.0)
        self.controller = massController() # testing controller
        self.reference = signalGenerator(amplitude=0.5, frequency=0.03, y_offset=0.5)
        self.disturbance = signalGenerator(amplitude=0.0)
        self.noise = signalGenerator(amplitude=0.0)

        if render_mode is 'human':
            # instantiate the simulation plots and animation
            self.dataPlot = dataPlotter()
            self.animation = massAnimation()

        # Intialize parameters
        self.t = P.t_start  # time starts at t_start # TODO this may mess things up
        self.F_max = P.F_max
        self.zdot_max = P.zdot_max
        self.z_max = P.z_max
        self.dt = P.Ts

        # Initialize action space
        # self.action_space = spaces.Box(
        #     low=-self.F_max, high=self.F_max, shape=(1,), dtype=np.float32
        # )
        # BEST PRACTICE: normalize action space to be between -1 and 1
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )

        high_observation = np.array([self.z_max, self.z_max], dtype=np.float32)
        low_observation = np.array([-self.z_max, -self.zdot_max], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=low_observation, high=high_observation, dtype=np.float32
        )

        # Initialize seed
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # print('Stepping...')
        # print('state:', self.state, 'action:', action, 'count:', self.count)
        # self.action_space.contains(np.array([18.0]).reshape(action.shape))
        # assert self.action_space.contains(
        #     action
        # ), f"{action!r}({type(action)}) invalid"

        # put normalized action back within bounds
        u = action * self.F_max

        u = action
        # check that u is within bounds
        u = np.clip(u, -self.F_max, self.F_max).item(0)
        self.last_u = u

        # Propagate dynamics in between plot samples
        self._run_dynamics(u)
        
        # get reference from signal generator
        # self.ref = self.reference.step(self.t)
        self.ref = 1.0

        # get state from mass
        z = self.mass.state.item(0)
        zdot = self.mass.state.item(1)

        # Work the magic by manipulating the cost function
        # my current philosophy is to increase costs on position, 
        # try to minimize velocity, and minimize input
        # loss function
        costs = 10*(z - self.ref)**2 + 0.01 * zdot ** 2 + 0.0001 * (u ** 2)

        terminated = bool(
            z < -self.z_max
            or z > self.z_max
            or zdot < -self.zdot_max
            or zdot > self.zdot_max
        )

        # Increment time
        self.t = self.t + self.dt

        if terminated is True:
            # print('Terminated')
            pass

        truncated = False
        if self.count > self.max_count:
            truncated = True
        else:
            self.count += 1
            truncated = False

        # if self.render_mode == 'human':
        #     self.render()
        return self._get_obs(), -costs, terminated, truncated, {}

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        # print('Resetting...')
        # reset dynamic state 
        super().reset(seed=seed)
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05
        )
        # high = np.array([self.z_max, self.zdot_max])
        # low = np.array([-self.z_max, -self.zdot_max])
        # rand = np.random.rand(2,)*.1
        # high = np.zeros((2,)) + rand
        # low = np.zeros((2,)) - rand
        self.state = self.np_random.uniform(low=low, high=high, size=(2,))
        self.mass.reset(self.state)
        self.last_u = None
        self.t = P.t_start
        self.count = 0
        self.render_count = 0
        # if self.render_mode == 'human':
        #     self.render()
        return self._get_obs(), {}

    def _get_obs(self):
        # print('Getting obs...')
        return np.array([
            self.mass.state.item(0),
            self.mass.state.item(1)
        ], dtype=np.float32)

    def _run_dynamics(self, u: float):
        # print('Running dynamics...')
        # Dynamics from custom simulator
        # u = np.ones((1,))*self.F_max
        y = self.mass.update(u)
        # self._set_state()
        self._set_state(self.mass.state.item(0), self.mass.state.item(1))

    def _set_state(self, z, zdot):
        # print('set_state...')
        self.state = np.array([z, zdot])

    def render(self):
        # print('Rendering...')
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        elif self.render_mode is 'human':
            if self.render_count > self.render_max_count:
                self.render_count = 0
                if self.animation is None:
                    self.animation = massAnimation()
                else:
                    # self.animation.update(self.arm.state)
                    self.animation.update(self.mass.state)

                    # the pause causes the figure to be displayed during the
                    # simulation
                    plt.pause(0.0001)

                if self.dataPlot is None:
                    self.dataPlot = dataPlotter()
                else:
                    self.dataPlot.update(self.t, self.ref, self.mass.state, self.last_u)
                    # the pause causes the figure to be displayed during the
                    # simulation
                    plt.pause(0.0001)
            else:
                self.render_count += 1
        else: 
            raise NotImplementedError

    def close(self):
        if self.animation:
            self.animation = None
        if self.dataPlot:
            self.dataPlot = None

def angle_normalize(z):
    return ((z + np.pi) % (2 * np.pi)) - np.pi