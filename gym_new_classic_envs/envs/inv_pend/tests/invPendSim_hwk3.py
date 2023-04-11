import matplotlib.pyplot as plt
import sys
sys.path.append('..')  # add parent directory
import gym_new_classic_envs.envs.inv_pend.inv_pend_resources.invPendParam as P
from gym_new_classic_envs.utils.signalGenerator import signalGenerator
from gym_new_classic_envs.envs.inv_pend.inv_pend_resources.invPendAnimation import InvertedPendulumAnimation
from gym_new_classic_envs.envs.inv_pend.inv_pend_resources.invPendDataPlotter import dataPlotter
from gym_new_classic_envs.envs.inv_pend.inv_pend_resources.invPendDynamics import InvertedPendulumDynamics

# instantiate pendulum, controller, and reference classes
pendulum = InvertedPendulumDynamics(alpha=0.0)
reference = signalGenerator(amplitude=0.5, frequency=0.02)
force = signalGenerator(amplitude=1, frequency=1)

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = InvertedPendulumAnimation()

t = P.t_start  # time starts at t_start
while t < P.t_end:  # main simulation loop

    # Propagate dynamics at rate Ts
    t_next_plot = t + P.t_plot
    while t < t_next_plot:
        r = reference.square(t)
        u = force.sin(t)
        y = pendulum.update(u)  # Propagate the dynamics
        t = t + P.Ts  # advance time by Ts

    # update animation and data plots at rate t_plot
    animation.update(pendulum.state)
    dataPlot.update(t, r, pendulum.state, u)

    # the pause causes figure to be displayed during simulation
    plt.pause(0.0001)

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
