#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

from gekko import GEKKO

import gym_new_classic_envs.envs.inv_pend.inv_pend_resources.invPendParam as P
from gym_new_classic_envs.envs.inv_pend.inv_pend_resources.invPendDynamics import InvertedPendulumDynamics
from gym_new_classic_envs.envs.inv_pend.inv_pend_controllers.invPendController import InvertedPendulumController
from gym_new_classic_envs.utils.signalGenerator import signalGenerator
from gym_new_classic_envs.envs.inv_pend.inv_pend_resources.invPendAnimation import InvertedPendulumAnimation
from gym_new_classic_envs.envs.inv_pend.inv_pend_resources.invPendDataPlotter import dataPlotter

# instantiate pendulum, controller, and reference classes
pendulum = InvertedPendulumDynamics()
controller = InvertedPendulumController()
reference = signalGenerator(amplitude=0.5, frequency=0.05)
disturbance = signalGenerator(amplitude=0)

############################################
# instantiate gekko model
############################################
m = GEKKO(remote=False)

#Defining the time parameter (0, 1)
N = 100
t = np.linspace(0,1,N)
# m.time = t
# m.time = [0,0.02,0.04,0.06,0.08,0.1,0.12,0.15,0.2]
# m.time = np.arange(0,5,0.2)
m.time = np.arange(0,4,0.01)

# Final Time
final = np.zeros_like(m.time)
final[-1] = 1
final = m.Param(final)

# Parameters
m1 = m.Param(value=P.m1)
m2 = m.Param(value=P.m2)
ell = m.Param(value=P.ell)
g = m.Param(value=P.g)
b = m.Param(value=P.b)

# MV
F = m.MV(value=0.0,lb=-P.F_max,ub=P.F_max)
F.STATUS = 1
# tau.DCOST = 0.01

# CV
theta = m.CV(value=P.theta0,lb=-10*np.pi/180.,ub=10*np.pi/180.)
theta.STATUS = 1
theta.FSTATUS = 1
theta.TR_INIT = 1
# theta.TAU = 0.01
theta.WSP = 1
# theta.COST = -18
thetadot = m.CV(value=P.thetadot0)
thetadot.STATUS = 1
thetadot.FSTATUS = 1
thetadot.TR_INIT = 1
z = m.CV(value=P.z0)
z.STATUS = 1
z.FSTATUS = 1
z.TR_INIT = 1
zdot = m.CV(value=P.zdot0)
zdot.STATUS = 1
zdot.FSTATUS = 1
zdot.TR_INIT = 1

# Dynamic Relationships
m.Equation(z.dt() == zdot)
m.Equation(theta.dt() == thetadot)
m.Equation((m1+m2)*zdot.dt() + m1*(ell/2.0)*m.cos(theta)*thetadot.dt() == \
           m1*(ell/2.0)*thetadot**2*m.sin(theta) + F - b*zdot)
m.Equation(m1*(ell/2.0)*m.cos(theta)*zdot.dt() + m1*(ell**2/3.0)*thetadot.dt() == \
           m1*g*(ell/2.0)*m.sin(theta))

# Global Options
m.options.CV_TYPE = 2
m.options.SOLVER = 3
m.options.IMODE = 6 
m.options.NODES = 3

############################################

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = InvertedPendulumAnimation()

t = P.t_start  # time starts at t_start
y = pendulum.h()  # output of system at start of simulation

while t < P.t_end:  # main simulation loop
    # Propagate dynamics in between plot samples
    t_next_plot = t + P.t_plot
    ################
    # MPC
    ################
    r = reference.square(t)
    n = 0.0  #noise.random(t)  # simulate sensor noise
    x = pendulum.state
    z.MEAS = x[0,0] + n
    theta.MEAS = x[1,0] + n
    zdot.MEAS = x[2,0] + N
    thetadot.MEAS = x[3,0] + n
    m.solve(disp=False)
    z.SP = r
    z.SPHI = r + 0.1
    z.SPLO = r - 0.1
    # retrieve new input values
    u = F.NEWVAL
    ################

    while t < t_next_plot:
        # r = reference.square(t)  # reference input
        d = disturbance.step(t)  # input disturbance
        # n = 0.0  #noise.random(t)  # simulate sensor noise
        # x = pendulum.state  # use state instead of output
        # u = controller.update(r, x)  # update controller
        y = pendulum.update(u + d)  # propagate system
        t = t + P.Ts  # advance time by Ts

    # update animation and data plots
    animation.update(pendulum.state)
    dataPlot.update(t, r, pendulum.state, u)
    plt.pause(0.0001)

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
