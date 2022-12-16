import numpy as np
from gym.envs.classic_control import PendulumEnv

class CustomPendulumEnv(PendulumEnv):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, g=10.0, target=0.0):
        super(CustomPendulumEnv, self).__init__()

        self.target = target

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        # costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)
        costs = (angle_normalize(th) - self.target) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)


        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l ** 2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi