import numpy as np
import gym_new_classic_envs.envs.mass.mass_controllers.state_feedback.pole_placement.massParamHW11 as P

class massController:
    # dirty derivatives to estimate thetadot
    def __init__(self):
        self.K = P.K  # state feedback gain
        self.kr = P.kr  # Input gain
        self.limit = P.F_max  # Maximum torque
        self.Ts = P.Ts  # sample rate of controller

    def update(self, z_r, x):
        z = x.item(0)
        zdot = x.item(1)

        # compute feedback linearizing force F_fl
        F_fl = P.k * z

        # copmute the state feedback controller
        F_tilde = -self.K @ x + self.kr * z_r

        # compute total torque
        # F = F_fl + F_tilde
        F = np.ones((1,1))*F_tilde
        F = self.saturate(F)
        return F

    def saturate(self,u):
        if abs(u) > self.limit:
            u = self.limit*np.sign(u)

        return u

