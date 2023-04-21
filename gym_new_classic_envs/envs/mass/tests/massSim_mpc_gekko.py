#!/usr/bin/env python3

from gekko import GEKKO

import matplotlib.pyplot as plt
import numpy as np
import time
import gym_new_classic_envs.envs.mass.mass_resources.massParam as P
from gym_new_classic_envs.envs.mass.mass_resources.massDynamics import massDynamics
from gym_new_classic_envs.envs.mass.mass_controllers.PD.massController import massController
from gym_new_classic_envs.utils.signalGenerator import signalGenerator
from gym_new_classic_envs.envs.mass.mass_resources.massAnimation import massAnimation
from gym_new_classic_envs.envs.mass.mass_resources.massDataPlotter import dataPlotter

# instantiate satellite, controller, and reference classes
massSys = massDynamics()
controller = massController()
reference = signalGenerator(amplitude=0.5, frequency=0.04)
disturbance = signalGenerator(amplitude=1.2)

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
m.time = np.arange(0,.6,0.01)

# Final Time
final = np.zeros_like(m.time)
final[-1] = 1
final = m.Param(final)

# Parameters
mass = m.Param(value=P.m)
k = m.Param(value=P.k)
b = m.Param(value=P.b)

# MV
F = m.MV(value=0.0,lb=-P.F_max,ub=P.F_max)
F.STATUS = 1
F.DCOST = 0.01

# CV
z = m.CV(value=P.z0)
z.STATUS = 1
z.FSTATUS = 1
z.TR_INIT = 1
z.TAU = 0.01
# z.WSP = 40
z.WSP = 120
# z.WSPHI = 1000
# theta.COST = -18
zdot = m.CV(value=P.zdot0)
zdot.STATUS = 1
zdot.FSTATUS = 1
zdot.TR_INIT = 1

# Dynamic Relationships
m.Equation(z.dt() == zdot)
m.Equation(zdot.dt() == (-k * z + F - b * zdot) / mass)

# Global Options
m.options.CV_TYPE = 2
m.options.SOLVER = 3
m.options.IMODE = 6 
m.options.NODES = 3

############################################

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = massAnimation()
t = P.t_start # time start at t_start
y = massSys.h() # output of system at start of simulation

time_in_control_loop = 0
count_in_ref = 0
num_secs_in_ref = 2
done = False

while t < P.t_end and not done: # main simulation loop
    # Propagate dynamics in between plot samples
    t_next_plot = t + P.t_plot
    start_time = time.time()
    ################
    # MPC
    ################
    r = reference.square(t)
    n = 0.0  #noise.random(t)  # simulate sensor noise
    x = massSys.state
    z.MEAS = x[0,0] + n
    zdot.MEAS = x[1,0] + n
    m.solve(disp=False)
    z.SP = r
    z.SPHI = r + 0.1
    z.SPLO = r - 0.1
    # print('SP', theta.SP)
    # retrieve new input values
    u = F.NEWVAL
    ################
    end_time = time.time()
    time_in_control_loop += end_time - start_time
    while t < t_next_plot: # updates control and dynamics at faster simulation rate
        # r = reference.square(t) # reference input
        d = disturbance.step(t) # input disturbance
        # n = 0.0 # noise.random(t) # simulate sensor noise
        # x = mass.state
        # u = controller.update(r, x) # update controller
        y = massSys.update(u + d) # propagate system
        # if abs(y-r) < abs(0.1*r):
        #     # print('within 10% of reference')
        #     count_in_ref += 1
        #     if count_in_ref > (num_secs_in_ref/P.Ts):
        #         print('within 10% of reference for {} iterations'.format(num_secs_in_ref))
        #         print(t)
        #         done = True
        # else:
        #     count_in_ref = 0
        t = t + P.Ts # advance time by Ts
    # update animation and data plots
    animation.update(massSys.state)
    dataPlot.update(t, r, massSys.state, u)
    plt.pause(0.0001) # the pause causes the figure to be displayed during the simulation

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = massAnimation()

t = P.t_start # time start at t_start
# Keeps the program from closing until the user presses a button
print('time in control loop: ', time_in_control_loop)
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
