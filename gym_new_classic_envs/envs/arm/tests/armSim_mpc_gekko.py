#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

from gekko import GEKKO

import gym_new_classic_envs.envs.arm.arm_resources.armParam as P
from gym_new_classic_envs.envs.arm.arm_resources.armDynamics import armDynamics
from gym_new_classic_envs.envs.arm.arm_controllers.armController import armController
from gym_new_classic_envs.utils.signalGenerator import signalGenerator
from gym_new_classic_envs.envs.arm.arm_resources.armAnimation import armAnimation
from gym_new_classic_envs.envs.arm.arm_resources.armDataPlotter import dataPlotter

# instantiate arm, controller, and reference classes
arm = armDynamics()
controller = armController()
reference = signalGenerator(amplitude=30*np.pi/180.0, frequency=0.05, y_offset=-30*np.pi/180.0)
disturbance = signalGenerator(amplitude=0.0)

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
m.time = np.arange(0,1,0.02)
print(m.time)

# Final Time
final = np.zeros_like(m.time)
final[-1] = 1
final = m.Param(final)

# Parameters
ell = m.Param(value=P.ell)
mass = m.Param(value=P.m)
g = m.Param(value=P.g)
# g = m.Param(value=0.0) 
# TODO: this is a problem. GEKKO isn't calculating the angle term correctly
b = m.Param(value=P.b)

# MV
print('tau_max', P.tau_max)
tau = m.MV(value=0.0,lb=-P.tau_max,ub=P.tau_max)
tau.STATUS = 1
# tau.DCOST = 0.01

# CV
theta = m.CV(value=P.theta0)
theta.STATUS = 1
theta.FSTATUS = 1
theta.TR_INIT = 1
# theta.COST = -18
thetadot = m.CV(value=P.thetadot0,lb=-P.thetadot_max,ub=P.thetadot_max)
thetadot.STATUS = 1
thetadot.FSTATUS = 1
thetadot.TR_INIT = 1
# x = m.CV(value=P.x0)
# x.STATUS = 1
# x.FSTATUS = 1
# x.TR_INIT = 1

# Dynamic Relationships
m.Equations([theta.dt() == thetadot])
m.Equations([thetadot.dt() == (3.0/mass/ell**2) * (tau - b*thetadot \
                                                - mass*g*ell/2.0*m.cos(theta))])

# Global Options
m.options.CV_TYPE = 2
m.options.SOLVER = 3
m.options.IMODE = 6 
m.options.NODES = 3

############################################

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = armAnimation()

t = P.t_start  # time starts at t_start
y = arm.h()  # output of system at start of simulation
while t < P.t_end:  # main simulation loop
    # Get referenced inputs from signal generators
    # Propagate dynamics in between plot samples
    t_next_plot = t + P.t_plot

    ################
    # MPC
    ################
    r = reference.square(t)
    n = 0.0  #noise.random(t)  # simulate sensor noise
    x = arm.state
    theta.MEAS = x[0,0] + n
    thetadot.MEAS = x[1,0] + n
    m.solve(disp=False)
    theta.SP = r
    # print('SP', theta.SP)
    # retrieve new input values
    u = tau.NEWVAL
    ################

    # updates control and dynamics at faster simulation rate
    while t < t_next_plot: 
        # r = reference.square(t)
        d = disturbance.step(t)  # input disturbance
        # n = 0.0  #noise.random(t)  # simulate sensor noise
        # x = arm.state
        # u = controller.update(r, x)  # update controller
        y = arm.update(u + d)  # propagate system
        t = t + P.Ts  # advance time by Ts

    # update animation and data plots
    animation.update(arm.state)
    dataPlot.update(t, r, arm.state, u)

    # the pause causes the figure to be displayed for simulation
    plt.pause(0.0001)  

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
