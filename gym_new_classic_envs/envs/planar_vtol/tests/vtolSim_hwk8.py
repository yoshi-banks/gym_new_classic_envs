import matplotlib.pyplot as plt
import sys

import numpy as np

# import os
# import sys
# sys.path.append(os.path.join(os.path.dirname(sys.path[0])))  # add parent directory
import gym_new_classic_envs.envs.planar_vtol.vtol_resources.VTOLParam as P
from gym_new_classic_envs.utils.signalGenerator import signalGenerator
from gym_new_classic_envs.envs.planar_vtol.vtol_resources.VTOLAnimation import VTOLAnimation
from gym_new_classic_envs.envs.planar_vtol.vtol_resources.VTOLDataPlotter import dataPlotter
from gym_new_classic_envs.envs.planar_vtol.vtol_resources.VTOLDynamics import VTOLDynamics
from gym_new_classic_envs.envs.planar_vtol.vtol_controllers.VTOLController import VTOLController

# instantiate satellite, controller, and reference classes
vtol = VTOLDynamics()
referenceH = signalGenerator(amplitude=5.0, frequency=0.01, y_offset=5)
referenceZ = signalGenerator(amplitude=2.5, frequency=.08, y_offset=3.0)
disturbance = signalGenerator(amplitude=0.0)
noise = signalGenerator(amplitude=1.0)

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = VTOLAnimation()
controller = VTOLController()

t = P.t_start  # time starts at t_start
while t < P.t_end:  # main simulation loop

    # Propagate dynamics in between plot samples
    t_next_plot = t + P.t_plot

    # updates control and dynamics at faster simulation rate
    while t < t_next_plot:  
        rH = referenceH.square(t)
        rZ = referenceZ.square(t)
        n = 0.0 #noise.random(t)
        x = vtol.state + n
        d = disturbance.step(t)
        u = np.array(controller.update(rH,rZ,x))
        y = vtol.update(u)  # Propagate the dynamics
        t = t + P.Ts  # advance time by Ts

    # update animation and data plots
    animation.update(vtol.state, rZ)
    dataPlot.update(t, vtol.state, rZ, rH, u.item(0), u.item(1))

    # the pause causes the figure to be displayed during the
    # simulation
    plt.pause(0.0001)  

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
