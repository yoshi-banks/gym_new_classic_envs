import numpy as np

import sys

sys.path.append('..')  # add parent directory
import gym_new_classic_envs.envs.planar_vtol.vtol_resources.VTOLParam as P
import gym_new_classic_envs.envs.planar_vtol.vtol_controllers.VTOLParamHW8 as P8

class VTOLController:
    def __init__(self):
        # Instantiates the PD object
        # self.kp_th = P.kptheta
        # self.kd_th = P.kdtheta
        # self.kp_z = P.kpz
        # self.kd_z = P.kdz
        # self.kp_h = P.kph
        # self.kd_h = P.kdh
        self.kp_h = P8.kp_h
        self.kd_h = P8.kd_h
        self.kp_z = P8.kp_z
        self.kd_z = P8.kd_z
        self.kp_th = P8.kp_th
        self.kd_th = P8.kd_th
        self.F_max = P.F_max

    def update(self, h_r, z_r, x):
        z = x.item(0)
        h = x.item(1)
        theta = x.item(2)
        zdot = x.item(3)
        hdot = x.item(4)
        thetadot = x.item(5)

        theta_e = 0.0
        F_e = ((P.mc + 2 * P.mr) * P.g)/np.cos(theta_e)

        F_tilde = self.kp_h * (h_r - h) - self.kd_h * hdot

        theta_r = self.kp_z * (z_r - z) - self.kd_z * zdot

        tau = self.kp_th * (theta_r - theta) - self.kd_th * thetadot

        F = F_tilde #F_e + F_tilde

        F = self.saturate(F)

        return F, tau

    def saturate(self, u):
        if abs(u) > self.F_max:
            u = self.F_max * np.sign(u)
        return u
