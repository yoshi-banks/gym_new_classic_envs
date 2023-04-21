import matplotlib.pyplot as plt
import numpy as np

import gym_new_classic_envs.envs.arm.arm_resources.armParam as P
from gym_new_classic_envs.envs.arm.arm_resources.armDynamics import armDynamics
from gym_new_classic_envs.envs.arm.arm_controllers.PD.armController import armController
from gym_new_classic_envs.utils.signalGenerator import signalGenerator
from gym_new_classic_envs.envs.arm.arm_resources.armAnimation import armAnimation
from gym_new_classic_envs.envs.arm.arm_resources.armDataPlotter import dataPlotter

# instantiate arm, controller, and reference classes
arm = armDynamics()
controller = armController()
reference = signalGenerator(amplitude=30*np.pi/180.0, frequency=0.05)
disturbance = signalGenerator(amplitude=0.0)

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = armAnimation()

t = P.t_start  # time starts at t_start
y = arm.h()  # output of system at start of simulation
while t < P.t_end:  # main simulation loop
    # Get referenced inputs from signal generators
    # Propagate dynamics in between plot samples
    t_next_plot = t + P.t_plot

    # updates control and dynamics at faster simulation rate
    while t < t_next_plot: 
        r = reference.square(t)
        d = disturbance.step(t)  # input disturbance
        n = 0.0  #noise.random(t)  # simulate sensor noise
        x = arm.state
        print('x:', x[0,0])
        u = controller.update(r, x)  # update controller
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
