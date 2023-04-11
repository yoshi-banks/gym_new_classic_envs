import matplotlib.pyplot as plt

import gym_new_classic_envs.envs.arm.arm_resources.armParam as P
from gym_new_classic_envs.utils.signalGenerator import signalGenerator
from gym_new_classic_envs.envs.arm.arm_resources.armAnimation import armAnimation
from gym_new_classic_envs.envs.arm.arm_resources.armDataPlotter import dataPlotter
from gym_new_classic_envs.envs.arm.arm_resources.armDynamics import armDynamics

# instantiate arm, controller, and reference classes
arm = armDynamics()
reference = signalGenerator(amplitude=0.01, frequency=0.02)
torque = signalGenerator(amplitude=0.2e2, frequency=0.05)

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = armAnimation()

t = P.t_start  # time starts at t_start
while t < P.t_end:  # main simulation loop
    # Propagate dynamics in between plot samples
    t_next_plot = t + P.t_plot

    # updates control and dynamics at faster simulation rate
    while t < t_next_plot:  

        # Get referenced inputs from signal generators
        r = reference.square(t)
        u = torque.square(t)
        y = arm.update(u)  # Propagate the dynamics
        t = t + P.Ts  # advance time by Ts
    # update animation and data plots
    animation.update(arm.state)
    dataPlot.update(t, r, arm.state, u)

    # the pause causes the figure to be displayed during the
    # simulation
    plt.pause(0.0001)  

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
