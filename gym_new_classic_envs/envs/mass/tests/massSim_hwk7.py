import sys
# sys.path.append('..') # add parent directory
import matplotlib.pyplot as plt
import numpy as np
import gym_new_classic_envs.envs.mass.mass_resources.massParam as P
from gym_new_classic_envs.envs.mass.mass_resources.massDynamics import massDynamics
from gym_new_classic_envs.envs.mass.mass_controllers.massController import massController
from gym_new_classic_envs.utils.signalGenerator import signalGenerator
from gym_new_classic_envs.envs.mass.mass_resources.massAnimation import massAnimation
from gym_new_classic_envs.envs.mass.mass_resources.massDataPlotter import dataPlotter

# instantiate satellite, controller, and reference classes
mass = massDynamics()
controller = massController()
reference = signalGenerator(amplitude=0.5, frequency=0.04)
disturbance = signalGenerator(amplitude=1.2)

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = massAnimation()
t = P.t_start # time start at t_start
y = mass.h() # output of system at start of simulation
while t < P.t_end: # main simulation loop
    # Propagate dynamics in between plot samples
    t_next_plot = t + P.t_plot
    while t < t_next_plot: # updates control and dynamics at faster simulation rate
        r = reference.square(t) # reference input
        d = disturbance.step(t) # input disturbance
        n = 0.0 # noise.random(t) # simulate sensor noise
        x = mass.state
        u = controller.update(r, x) # update controller
        y = mass.update(u + d) # propagate system
        t = t + P.Ts # advance time by Ts
    # update animation and data plots
    animation.update(mass.state)
    dataPlot.update(t, r, mass.state, u)
    plt.pause(0.0001) # the pause causes the figure to be displayed during the simulation

# Keeps the program from closing until the user presses a button
print('Press key to close')
plt.waitforbuttonpress()
plt.close()