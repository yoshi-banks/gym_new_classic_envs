import numpy as np
import gym_new_classic_envs.envs.arm.arm_controllers.PID.armParamHW10 as P
import sys
sys.path.append('..')  # add parent directory
import gym_new_classic_envs.envs.arm.arm_resources.armParam as P0
from gym_new_classic_envs.envs.arm.arm_controllers.PID.PIDControl import PIDControl


class armController:

    def __init__(self):
        # Instantiates the PD object
        self.thetaCtrl = PIDControl(P.kp, P.ki, P.kd,
                                    P0.tau_max, P.beta, P.Ts)
        self.limit = P0.tau_max

    def update(self, theta_r, y):
        theta = y.item(0)

        # compute feedback linearized torque tau_fl
        tau_fl = P0.m * P0.g * (P0.ell / 2.0) * np.cos(theta)

        # compute the linearized torque using PD
        tau_tilde = self.thetaCtrl.PID(theta_r, theta, False)
        # tau_tilde = self.

        # compute total torque
        tau = tau_fl + tau_tilde
        tau = self.saturate(tau)
        return tau

    def saturate(self, u):
        if abs(u) > self.limit:
            u = self.limit*np.sign(u)
        return u







