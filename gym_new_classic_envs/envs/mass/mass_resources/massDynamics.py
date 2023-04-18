import numpy as np
import random
import gym_new_classic_envs.envs.mass.mass_resources.massParam as P

class massDynamics:
    def __init__(self, alpha=0.0):
        # Initial state conditions
        self.state = np.array([
            [P.z0],  # initial position
            [P.zdot0]  # initial velocity
        ])

        # simulation time step
        self.Ts = P.Ts

        # mass of box
        self.m = P.m * (1.+alpha*(2.*np.random.rand()-1.))

        # spring coefficient
        self.k = P.k * (1.+alpha*(2.*np.random.rand()-1.))

        # Damping coefficient, Ns
        self.b = P.b * (1.+alpha*(2.*np.random.rand()-1.))
        self.force_limit = P.F_max

    def reset(self, state):
        # Reset the state of the system to the initial conditions
        assert len(state) == 2
        self.state = state.reshape((2, 1))

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
        zdot = state.item(1)
        force = u

        M = self.m

        C = -self.k * z + force - self.b * zdot


        #tmp = np.linalg.inv(M) @ C
        zddot = C/M

        # build xdot and return
        xdot = np.array([[zdot], [zddot]])
        return xdot

    def h(self):
        # return y = h(x)
        z = self.state.item(0)
        y = np.array([[z]])

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
            u = limit*np.sign(u)
        return u
