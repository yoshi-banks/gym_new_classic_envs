import matplotlib.ft2font
import numpy as np
import gym_new_classic_envs.envs.mass.mass_controllers.PID.massParamHW10 as P10
import sys
sys.path.append('..')
import gym_new_classic_envs.envs.mass.mass_resources.massParam as P
from gym_new_classic_envs.envs.mass.mass_controllers.PID.PIDControl import PIDControl

class massController:

    def __init__(self):
        self.zCtrl = PIDControl(P10.kp, P10.ki, P10.kd,
                                P.F_max, P.beta, P.Ts)
        # self.kp = P10.kp
        # self.kd = P10.kd
        self.F_max = P.F_max

    def update(self, z_r, y):
        z = y.item(0)

        # compute feedback linearized force F_fl
        F_fl = P.k * z

        # compute the linearize force using PID
        F_tilde = self.zCtrl.PID(z_r, z, False)

        # compute the total force
        F = F_fl + F_tilde
        F = self.saturate(F, self.F_max)
        return F

    def saturate(self, u, limit):
        if abs(u) > limit:
            u = limit * np.sign(u)
        return u