import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))  # add parent directory
from gym_new_classic_envs.envs.satellite.satellite_resources import satelliteParam as P
from gym_new_classic_envs.utils.signalGenerator import signalGenerator
from gym_new_classic_envs.envs.satellite.satellite_resources.satelliteAnimation import satelliteAnimation
from gym_new_classic_envs.envs.satellite.satellite_resources.satelliteDataPlotter import dataPlotter
from gym_new_classic_envs.envs.satellite.satellite_resources.satelliteDynamics import satelliteDynamics


# instantiate satellite, controller, and reference classes
satellite = satelliteDynamics()
reference = signalGenerator(amplitude=0.5, frequency=0.1)
torque = signalGenerator(amplitude=0.1, frequency=0.1)

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = satelliteAnimation()

t = P.t_start  # time starts at t_start
while t < P.t_end:  # main simulation loop

    # Propagate dynamics in between plot samples
    t_next_plot = t + P.t_plot

    # updates control and dynamics at faster simulation rate
    while t < t_next_plot:  
        r = reference.square(t)
        u = torque.sin(t)
        y = satellite.update(u)  # Propagate the dynamics
        t = t + P.Ts  # advance time by Ts

    # update animation and data plots
    animation.update(satellite.state)
    dataPlot.update(t, r, satellite.state, u)

    # the pause causes the figure to be displayed during the
    # simulation
    plt.pause(0.0001)  

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
