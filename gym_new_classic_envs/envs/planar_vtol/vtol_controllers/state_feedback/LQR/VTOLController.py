import numpy as np
import gym_new_classic_envs.envs.planar_vtol.vtol_controllers.state_feedback.LQR.VTOLParamHW11 as P

class VTOLController:
    def __init__(self):
        self.K_lat = P.K_lat  # state feedback gain
        self.K_lon = P.K_lon
        self.kr_lat = P.kr_lat
        self.kr_lon = P.kr_lon
        self.tau_max = P.tau_max  # Maxiumum torque
        self.F_max = P.F_max
        self.Ts = P.Ts  # sample rate of controller

    def update(self, rH, rZ, x):
        # print('K_lat: ', self.K_lat)
        # print('K-lon: ', self.K_lon)
        # print('x: ', x)
        # print('testing: ', x[:1])
        # print('testing.shape', x[:1].shape)
        # print('testing2: ', x[2:4])
        # print('testing2.shape: ', x[2:4].shape)
        # print('testing3: ', x[5:])
        # print('testing3.shape', x[5:].shape)
        # print('testing4: ', np.concatenate((x[:1], x[2:4], x[5:]),axis=0))
        # print('testing5: ', np.concatenate((x[1:2], x[4:5]),axis=0))
        x_lon = np.concatenate((x[1:2], x[4:5]), axis=0)
        x_lat = np.concatenate((x[:1], x[2:4], x[5:]), axis=0)
        # print('x_lon: ', x_lon)
        # print('x_lat: ', x_lat)
        # print('kr_lon: ', self.kr_lon)
        # print('kr_lat: ', self.kr_lat)

        # Compute the state feedback longitude controller
        F_unsat = -self.K_lon @ x_lon + self.kr_lon * rH
        # F_unsat = P.Fe #+ F_tilde
        # print('F_unsat', F_unsat)
        F = self.saturate(F_unsat.item(0), self.F_max)

        # Compute the state feedback latitude controller
        tau_unsat = -self.K_lat @ x_lat + self.kr_lat * rZ
        tau = self.saturate(tau_unsat.item(0), self.tau_max)

        return [F, tau]

    def saturate(self,u,limit):
        if abs(u) > limit:
            u = limit*np.sign(u)
        return u

