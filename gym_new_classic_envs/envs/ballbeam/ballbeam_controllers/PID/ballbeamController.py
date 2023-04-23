import numpy as np

import gym_new_classic_envs.envs.ballbeam.ballbeam_resources.ballbeamParam as P
import gym_new_classic_envs.envs.ballbeam.ballbeam_controllers.PID.ballbeamParamHW10 as P10
from gym_new_classic_envs.envs.ballbeam.ballbeam_controllers.PID.PIDControl import PIDControl

class ballbeamController:
    def __init__(self):
        #instantiates the SS_ctrl object
        self.zCtrl = PIDControl(P10.kp_z, P10.ki_z, P10.kd_z,
                                P10.theta_max, P.sigma, P.Ts)
        self.thetaCtrl = PIDControl(P10.kp_th, 0.0, P10.kd_th,
                                    P10.F_max, P.sigma, P.Ts)
        # self.kp_z = P10.kp_z
        # self.kd_z = P10.kd_z
        # self.kp_th = P10.kp_th
        # self.kd_th = P10.kd_th
        #self.theta_max = P8.theta_max
        self.F_max = P.F_max

    def update(self, z_r, y):
        z = y.item(0)
        theta = y.item(1)

        # the reference angle for theta comes from the outer loop PID control
        theta_r = self.zCtrl.PID(z_r, z, flag=False)

        # the force applied to the beam comes from the inner loop PD ontrol
        F_tilde = self.thetaCtrl.PD(theta_r, theta, flag=False)

        # z = state.item(0)
        # theta = state.item(1)
        # zdot = state.item(2)
        # thetadot = state.item(3)
        #
        # theta_r = self.kp_z * (z_r - z) - self.kd_z * zdot
        #
        # F_tilde = self.kp_th * (theta_r - theta) - self.kd_th * thetadot
        #
        # feedback linearizing force
        F_fl = P.m1 * P.g * (z / P.length) + P.m2 * P.g / 2.0

        # total force
        F = F_tilde + F_fl

        F = self.saturate(F, self.F_max)
        return F

    def saturate(self, u, limit):
        if abs(u) > limit:
            u = limit * np.sign(u)
        return u
