import numpy as np 
import random
import gym_new_classic_envs.envs.planar_vtol.vtol_resources.VTOLParam as P
import math as m

class VTOLDynamics:
    def __init__(self, alpha=0.0):
        # Initial state conditions
        self.state = np.array([
            [P.z0],  # initial base angle
            [P.h0],  # initial panel angle
            [P.theta0],  # initial angular velocity of base
            [P.zdot0],
            [P.hdot0],
            [P.thetadot0] # initial angular velocity of panel
        ])

        # simulation time step
        self.Ts = P.Ts
        self.mc = P.mc * (1+2*alpha*np.random.rand() - alpha)
        self.mr = P.mr * (1+2*alpha*np.random.rand() - alpha)
        self.Jc = P.Jc * (1+2*alpha*np.random.rand() - alpha)
        self.d = P.d * (1+2*alpha*np.random.rand() - alpha)
        self.mu = P.mu * (1+2*alpha*np.random.rand() - alpha)
        self.F_wind = P.F_wind * (1+2*alpha*np.random.rand() - alpha)

        self.force_limit = P.F_max

    def update(self, u):
        # This is the external method that takes the input u at time
        # t and returns the output y at time t.
        # saturate the input torque
        #u = self.saturate(u, self.force_limit)

        self.rk4_step(u)  # propagate the state by one time sample
        y = self.h()  # return the corresponding output

        return y

    def f(self, state, u):
        # Return xdot = f(x,u)
        z = state.item(0)
        h = state.item(1)
        theta = state.item(2)
        zdot = state.item(3)
        hdot = state.item(4)
        thetadot = state.item(5)
        F = u.item(0)
        tau = u.item(1)

        fr = .5 * F + tau / (2 * P.d)
        fl = .5 * F - tau / (2 * P.d)

        F_e = (P.mc + 2 * P.mr) * P.g

        # The equations of motion.
        zddot = (-(fr + fl) * np.sin(theta) + -self.mu * zdot +
                 self.F_wind) / (self.mc + 2.0 * self.mr)
        zddot = (-F_e * theta - P.mu * zdot)/(P.mc + 2 * P.mr)

        hddot = (-(self.mc + 2.0 * self.mr) * P.g * (fr + fl) * np.cos(
            theta)) / (self.mc + 2.0 * self.mr)
        hddot = F/(P.mc + 2 * P.mr)

        thetaddot = self.d * (fr - fl) / (self.Jc + 2.0 * self.mr *
                                          (self.d **2))
        thetaddot = tau/(P.Jc + 2 * P.mr * P.d ** 2)

        # build xdot and return
        xdot = np.array([[zdot], [hdot], [thetadot],
                         [zddot], [hddot], [thetaddot]])
        return xdot

    def h(self):
        # return y = h(x)
        z = self.state.item(0)
        h = self.state.item(1)
        theta = self.state.item(2)
        y = np.array([[z], [h], [theta]])

        return y

    def rk4_step(self, u):
        # Integrate ODE using Runge-Kutta RK4 algorithm
        F1 = self.f(self.state, u)
        F2 = self.f(self.state + self.Ts / 2 * F1, u)
        F3 = self.f(self.state + self.Ts / 2 * F2, u)
        F4 = self.f(self.state + self.Ts * F3, u)
        self.state += self.Ts / 6 * (F1 + 2 * F2 + 2 * F3 + F4)

    def saturate(self, u, limit):
        if abs(u) > limit:
            u = limit*np.sign(u)
        return u
