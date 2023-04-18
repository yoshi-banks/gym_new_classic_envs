import numpy as np

import sys

sys.path.append('..')  # add parent directory
import gym_new_classic_envs.envs.planar_vtol.vtol_resources.VTOLParam as P
import gym_new_classic_envs.envs.planar_vtol.vtol_controllers.PID.VTOLParamHW10 as P10
from gym_new_classic_envs.envs.planar_vtol.vtol_controllers.PID.PIDControl import PIDControl


class vtolController:
    def __init__(self):
        # Instantiates the PD object
        # self.kp_h = P10.kp_h
        # self.kd_h = P10.kd_h
        # self.kp_z = P10.kp_z
        # self.kd_z = P10.kd_z
        # self.kp_th = P10.kp_th
        # self.kd_th = P10.kd_th
        self.F_max = P.F_max
        # INstantiates the SS_ctrol object
        self.hCtrl = PIDControl(P10.kp_h, P10.ki_h, P10.kd_h,
                                P.F_max, P.beta, P.Ts)
        self.thetaCtrl = PIDControl(P10.kp_th, 0.0, P10.kd_th,
                             P10.theta_max, P.beta, P.Ts)
        self.zCtrl = PIDControl(P10.kp_z, P10.ki_z, P10.kd_z,
                                P10.tau_max, P.beta, P.Ts)

    def update(self, h_r, z_r, y):
        # use the state method
        # z = x.item(0)
        # h = x.item(1)
        # theta = x.item(2)
        # zdot = x.item(3)
        # hdot = x.item(4)
        # thetadot = x.item(5)
        #
        # theta_e = 0.0
        # F_e = ((P.mc + 2 * P.mr) * P.g)/np.cos(theta_e)
        # F_tilde = self.kp_h * (h_r - h) - self.kd_h * hdot
        # theta_r = self.kp_z * (z_r - z) - self.kd_z * zdot
        # tau = self.kp_th * (theta_r - theta) - self.kd_th * thetadot
        # F = F_tilde #F_e + F_tilde
        # F = self.saturate(F)

        # Use the sensors
        z = y.item(0)
        h = y.item(1)
        theta = y.item(2)
        #Equilibrium force
        F_e = ((P.mc + 2 * P.mr) * P.g)

        # the reference angle for theta comes from the outer loop PID control
        theta_r = self.zCtrl.PID(z_r, z, flag=False)

        # the torque applied to the drone comes from the horizontal inner loop PD control
        tau = self.thetaCtrl.PID(theta_r, theta, flag=False)

        # the force comes from the altitude control
        F_tilde = self.hCtrl.PID(h_r, h, flag=False)

        F = F_tilde

        # F = self.saturate(F)

        # the logitudinal force

        return F, tau

    def saturate(self, u):
        if abs(u) > self.F_max:
            u = self.F_max * np.sign(u)
        return u
