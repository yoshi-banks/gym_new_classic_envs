import numpy as np 
import random
import gym_new_classic_envs.envs.arm.arm_resources.armParam as P


class armDynamics:
    def __init__(self, alpha=0.0, theta0=P.theta0, thetadot0=P.thetadot0, 
                 m=P.m, ell=P.ell, b=P.b, g=P.g, Ts=P.Ts, tau_max=P.tau_max):

        # Set the arm to the inital state
        self.reset(theta0, thetadot0)

        # Mass of the arm, kg
        self.m = m * (1.+alpha*(2.*np.random.rand()-1.))

        # Length of the arm, m
        self.ell = ell * (1.+alpha*(2.*np.random.rand()-1.))

        # Damping coefficient, Ns
        self.b = b * (1.+alpha*(2.*np.random.rand()-1.))  

        # the gravity constant is well known, so we don't change it.
        self.g = g

        # sample rate at which the dynamics are propagated
        self.Ts = Ts  
        self.torque_limit = tau_max

    def update(self, u):
        # This is the external method that takes the input u at time
        # t and returns the output y at time t.
        # saturate the input torque
        u = self.saturate(u, self.torque_limit)
        
        self.rk4_step(u)  # propagate the state by one time sample
        # wrap theta to keep it between 0-2pi
        # self.state[0] = self.wrap(self.state[0], np.pi)
        y = self.h()  # return the corresponding output

        return y

    def f(self, state, tau):
        # Return xdot = f(x,u), the system state update equations
        # re-label states for readability
        theta = state.item(0)
        thetadot = state.item(1)
        thetaddot = (3.0/self.m/self.ell**2) * \
                    (tau - self.b*thetadot \
                     - self.m*self.g*self.ell/2.0*np.cos(theta))
        xdot = np.array([[thetadot],
                         [thetaddot]])
        
        return xdot

    def h(self):
        # return the output equations
        # could also use input u if needed
        theta = self.state.item(0)
        y = np.array([[theta]])

        return y

    def rk4_step(self, u):
        # Integrate ODE using Runge-Kutta RK4 algorithm
        F1 = self.f(self.state, u)
        F2 = self.f(self.state + self.Ts / 2 * F1, u)
        F3 = self.f(self.state + self.Ts / 2 * F2, u)
        F4 = self.f(self.state + self.Ts * F3, u)
        self.state += self.Ts / 6 * (F1 + 2 * F2 + 2 * F3 + F4)

    def reset(self, theta, thetadot):
        # Initial state conditions
        self.state = np.array([
            [theta],      
            [thetadot]
        ])  

    def saturate(self, u, limit):
        # TODO np.clip does this same thing
        if abs(u) > limit:
            u = limit*np.sign(u)

        return u

    def wrap(self, chi_1, chi_2):
        while chi_1 - chi_2 > np.pi:
            chi_1 = chi_1 - 2.0 * np.pi
        while chi_1 - chi_2 < -np.pi:
            chi_1 = chi_1 + 2.0 * np.pi
        return chi_1
