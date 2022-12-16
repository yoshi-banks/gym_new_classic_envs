import sys
sys.path.append('..')  # add parent directory
import matplotlib.pyplot as plt
import gym_new_classic_envs.envs.inv_pend.inv_pend_resources.invPendParam as P
from gym_new_classic_envs.envs.inv_pend.inv_pend_resources.invPendDynamics import InvertedPendulumDynamics
from gym_new_classic_envs.envs.inv_pend.inv_pend_controllers.invPendController import InvertedPendulumController
from gym_new_classic_envs.utils.signalGenerator import signalGenerator
from gym_new_classic_envs.envs.inv_pend.inv_pend_resources.invPendAnimation import InvertedPendulumAnimation
from gym_new_classic_envs.utils.dataPlotter import dataPlotter

# instantiate pendulum, controller, and reference classes
pendulum = InvertedPendulumDynamics()
controller = InvertedPendulumController()
reference = signalGenerator(amplitude=0.5, frequency=0.05)
disturbance = signalGenerator(amplitude=0)

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = InvertedPendulumAnimation()

t = P.t_start  # time starts at t_start
y = pendulum.h()  # output of system at start of simulation

while t < P.t_end:  # main simulation loop
    # Propagate dynamics in between plot samples
    t_next_plot = t + P.t_plot

    while t < t_next_plot:
        r = reference.square(t)  # reference input
        d = disturbance.step(t)  # input disturbance
        n = 0.0  #noise.random(t)  # simulate sensor noise
        x = pendulum.state  # use state instead of output
        u = controller.update(r, x)  # update controller
        y = pendulum.update(u + d)  # propagate system
        t = t + P.Ts  # advance time by Ts

    # update animation and data plots
    animation.update(pendulum.state)
    dataPlot.update(t, r, pendulum.state, u)
    plt.pause(0.0001)

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
