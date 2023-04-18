import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import matplotlib.pyplot as plt
import numpy as np
import gym_new_classic_envs.envs.mass.mass_resources.massParam as P
from gym_new_classic_envs.envs.mass.mass_resources.massDynamics import massDynamics
from gym_new_classic_envs.envs.mass.mass_controllers.PID.massController import massController
from gym_new_classic_envs.utils.signalGenerator import signalGenerator
from gym_new_classic_envs.envs.mass.mass_resources.massAnimation import massAnimation
from gym_new_classic_envs.envs.mass.mass_resources.massDataPlotter import dataPlotter

# instantiate mass, controller, and reference classes
mass = massDynamics(alpha=0.0)
controller = massController()
reference = signalGenerator(amplitude=0.5, frequency=0.03, y_offset=0.5)
disturbance = signalGenerator(amplitude=0.0)
noise = signalGenerator(amplitude=0.0)

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = massAnimation()
t = P.t_start # time starts at t_start
y = mass.h() # output of system at start of simulation
while t < P.t_end:
    # Propagate dynamics in between plot samples
    t_next_plot = t + P.t_plot
    while t < t_next_plot: # updates control and dynamics at faster simluation rate
        r = reference.square(t) # reference input
        d = disturbance.step(t) # input disturbance
        n = noise.random(t) # simulate sensor noise
        # x = mass.state
        u = controller.update(r, y + n) # update controller
        y = mass.update(u + d) # propagate
        t = t + P.Ts # advance time by Ts
    # update animation and data plots
    animation.update(mass.state)
    dataPlot.update(t, r, mass.state, u)
    plt.pause(0.0001)

# Keeps the program from closing until the user presses a button
print('Press key to close')
plt.waitforbuttonpress()
plt.close()