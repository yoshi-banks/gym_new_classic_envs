import numpy as np
import gym_new_classic_envs.envs.planar_vtol.vtol_controllers.state_feedback.LQR_integrator.VTOLParamHW12 as P

class VTOLController:
    def __init__(self):
        self.integrator_lon = 0.0
        self.integrator_lat = 0.0
        self.error_d1_lon = 0.0
        self.error_d1_lat = 0.0
        self.K_lat = P.K_lat  # state feedback gain
        self.K_lon = P.K_lon
        self.ki_lat = P.ki_lat
        self.ki_lon = P.ki_lon
        self.tau_max = P.tau_max  # Maxiumum torque
        self.F_max = P.F_max
        self.Ts = P.Ts  # sample rate of controller

    def update(self, rH, rZ, x):
        z = x.item(0)
        h = x.item(1)
        theta = x.item(2)
        zdot = x.item(3)
        hdot = x.item(4)
        thetadot = x.item(5)

        x_lon = np.array([[h],
                          [hdot]])
        x_lat = np.array([[z],
                          [theta],
                          [zdot],
                          [thetadot]])

        # integrate error
        error_lon = rH - h
        self.integrateError_lon(error_lon)

        error_lat = rZ - z
        self.integrateErorr_lat(error_lat)

        # Compute the state feedback controller

        # Compute the state feedback longitude controller
        F_unsat = -self.K_lon @ x_lon - self.ki_lon*self.integrator_lon

        F = self.saturate(F_unsat.item(0), self.F_max)

        # Compute the state feedback latitude controller
        tau_unsat = -self.K_lat @ x_lat - self.ki_lat*self.integrator_lat
        tau = self.saturate(tau_unsat.item(0), self.tau_max)

        return [F, tau]

    def integrateError_lon(self, error_lon):
        self.integrator_lon = self.integrator_lon + (self.Ts / 2.0) \
                              * (error_lon + self.error_d1_lon)

        self.error_d1_lon = error_lon

    def integrateErorr_lat(self, error_lat):
        self.integrator_lat = self.integrator_lat + (self.Ts / 2.0) \
                             * (error_lat + self.error_d1_lat)

        self.error_d1_lat = error_lat


    def saturate(self,u,limit):
        if abs(u) > limit:
            u = limit*np.sign(u)
        return u

