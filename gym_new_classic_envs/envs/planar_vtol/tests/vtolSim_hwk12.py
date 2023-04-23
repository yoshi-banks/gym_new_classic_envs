import sys

import matplotlib.pyplot as plt
import numpy as np
import gym_new_classic_envs.envs.planar_vtol.vtol_resources.VTOLParam as P
from gym_new_classic_envs.envs.planar_vtol.vtol_resources.VTOLDynamics import VTOLDynamics
from gym_new_classic_envs.envs.planar_vtol.vtol_controllers.state_feedback.LQR_integrator.VTOLController import VTOLController
from gym_new_classic_envs.utils.signalGenerator import signalGenerator
from gym_new_classic_envs.envs.planar_vtol.vtol_resources.VTOLAnimation import VTOLAnimation
from gym_new_classic_envs.envs.planar_vtol.vtol_resources.VTOLDataPlotter import dataPlotter

# instantiate satellite, controller, and reference classes
VTOL = VTOLDynamics(alpha=0.)
controller = VTOLController()
referenceH = signalGenerator(amplitude=5.0, frequency=0.05, y_offset=5.0)
referenceZ = signalGenerator(amplitude=5.0, frequency=0.05, y_offset=5.0)
disturbance = signalGenerator(amplitude=0.0)
noise = signalGenerator(amplitude=0.)

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = VTOLAnimation()

t = P.t_start  # time starts at t_start
y = VTOL.h()  # output of system at start of simulation
while t < P.t_end:  # main simulation loop
    # Propagate dynamics in between plot samples
    t_next_plot = t + P.t_plot
    while t < t_next_plot:  # updates control and dynamics at faster simulation rate
        rH = referenceH.square(t)  # reference input
        rZ = referenceZ.square(t)
        d = disturbance.step(t)  # input disturbance
        n = noise.random(t)  # simulate sensor noise
        x = VTOL.state
        u = np.array(controller.update(rH, rZ, x + n))  # update controller
        y = VTOL.update(u + d)  # propagate system
        t = t + P.Ts  # advance time by Ts
    # update animation and data plots
    animation.update(VTOL.state, rZ)
    dataPlot.update(t, VTOL.state, rZ, rH, u.item(0), u.item(1))
    plt.pause(0.0001)  # the pause causes the figure to be displayed during the simulation

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
