import numpy as np
import math as m
import gym_new_classic_envs.envs.ballbeam.ballbeam_resources.ballbeamParam as P

class ballbeamDynamics:
    def __init__(self, alpha=0.0):
        # Initial state conditions
        self.state = np.array([
            [P.z0],  # initial base angle
            [P.theta0],  # initial panel angle
            [P.zdot0],  # initial angular velocity of base
            [P.thetadot0],  # initial angular velocity of panel
        ])

        # simulation time step
        self.Ts = P.Ts

        # masses
        self.m1 = P.m1 * (1. + alpha * (2. * np.random.rand() - 1.))
        self.m2 = P.m2 * (1. + alpha * (2. * np.random.rand() - 1.))

        # length
        self.l = P.length * (1. + alpha * (2. * np.random.rand() - 1.))

        self.g = P.g

        self.force_limit = P.F_max

    def update(self, u):
        # This is the external method that takes the input u at time
        # t and returns the output y at time t.
        # saturate the input torque
        u = self.saturate(u, self.force_limit)

        self.rk4_step(u)  # propagate the state by one time sample
        y = self.h()  # return the corresponding output

        return y

    def f(self, state, u):
        # Return xdot = f(x,u)
        z = state.item(0)
        theta = state.item(1)
        zdot = state.item(2)
        thetadot = state.item(3)
        F = u
        # The equations of motion.
        zddot = (1.0/self.m1)*(self.m1*z*thetadot**2
                               - self.m1*P.g*np.sin(theta))
        thetaddot = (1.0/((self.m2*self.l**2)/3.0
                          + self.m1*z**2))*(-2.0*self.m1*z*zdot*thetadot
                                            - self.m1*P.g*z*np.cos(theta)
                                            - self.m2*P.g*self.l/2.0*np.cos(theta)
                                            + self.l*F*np.cos(theta))

        ##
        # thetaddot = (1.0/((self.m2*self.l**2)/2.0
        #                   + self.m1*z**2))*(-2.0*self.m1*z*zdot*thetadot
        #                                     - self.m1*P.g*z*np.cos(theta)
        #                                     - self.m2*P.g*self.l/2.0*np.cos(theta)
        #                                     + self.l*tau*np.cos(theta))
        ##



        # build xdot and return
        xdot = np.array([[zdot], [thetadot], [zddot],
                         [thetaddot]])
        return xdot

    def h(self):
        # return y = h(x)
        z = self.state.item(0)
        theta = self.state.item(1)
        y = np.array([[z], [theta]])

        return y

    def rk4_step(self, u):
        # Integrate ODE using Runge-Kutta RK4 algorithm
        F1 = self.f(self.state, u)
        F2 = self.f(self.state + self.Ts / 2 * F1, u)
        F3 = self.f(self.state + self.Ts / 2 * F2, u)
        F4 = self.f(self.state + self.Ts * F3, u)
        self.state = self.state + self.Ts / 6 * (F1 + (2 * F2) + (2 * F3) + F4)

    def saturate(self, u, limit):
        if abs(u) > limit:
            u = limit * np.sign(u)
        return u
