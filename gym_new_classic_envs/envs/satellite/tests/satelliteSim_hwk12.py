import sys
sys.path.append('..')  # add parent directory
import os
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))  # add parent directory
import matplotlib.pyplot as plt
import numpy as np
import gym_new_classic_envs.envs.satellite.satellite_resources.satelliteParam as P
from gym_new_classic_envs.envs.satellite.satellite_resources.satelliteDynamics import satelliteDynamics
from gym_new_classic_envs.envs.satellite.satellite_controllers.state_feedback.pole_placement.satelliteController import satelliteController
from gym_new_classic_envs.utils.signalGenerator import signalGenerator
from gym_new_classic_envs.envs.satellite.satellite_resources.satelliteAnimation import satelliteAnimation
from gym_new_classic_envs.envs.satellite.satellite_resources.satelliteDataPlotter import dataPlotter

# instantiate satellite, controller, and reference classes
satellite = satelliteDynamics()
controller = satelliteController()
reference = signalGenerator(amplitude=15.0*np.pi/180.0,
                            frequency=0.02)
disturbance = signalGenerator(amplitude=1.0)

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = satelliteAnimation()

t = P.t_start  # time starts at t_start
y = satellite.h()  # output of system at start of simulation
while t < P.t_end:  # main simulation loop
    # Propagate dynamics in between plot samples
    t_next_plot = t + P.t_plot
    while t < t_next_plot:  # updates control and dynamics at faster simulation rate
        r = reference.square(t)  # reference input
        d = disturbance.step(t)  # input disturbance
        n = 0.0  #noise.random(t)  # simulate sensor noise
        x = satellite.state
        u = controller.update(r, x)  # update controller
        y = satellite.update(u + d)  # propagate system
        t = t + P.Ts  # advance time by Ts
    # update animation and data plots
    animation.update(satellite.state)
    dataPlot.update(t, r, satellite.state, u)
    plt.pause(0.0001)  # the pause causes the figure to be displayed during the simulation

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
