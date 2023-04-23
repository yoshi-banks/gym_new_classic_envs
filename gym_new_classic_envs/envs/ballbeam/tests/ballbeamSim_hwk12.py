import sys
sys.path.append('..')  # add parent directory
import matplotlib.pyplot as plt
import gym_new_classic_envs.envs.ballbeam.ballbeam_resources.ballbeamParam as P
from gym_new_classic_envs.utils.signalGenerator import signalGenerator
from gym_new_classic_envs.envs.ballbeam.ballbeam_resources.ballbeamAnimation import ballbeamAnimation
from gym_new_classic_envs.envs.ballbeam.ballbeam_resources.ballbeamDataPlotter import dataPlotter
from gym_new_classic_envs.envs.ballbeam.ballbeam_resources.ballbeamDynamics import ballbeamDynamics
from gym_new_classic_envs.envs.ballbeam.ballbeam_controllers.state_feedback.LQR.ballbeamController import ballbeamController

# instantiate pendulum, controller, and reference classes
ballbeam = ballbeamDynamics(alpha=0.2)
controller = ballbeamController()
reference = signalGenerator(amplitude=0.5, frequency=0.05)
disturbance = signalGenerator(amplitude=10.25)

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = ballbeamAnimation()

t = P.t_start  # time starts at t_start
y = ballbeam.h()  # output of system at start of simulation
while t < P.t_end:  # main simulation loop
    # Propagate dynamics in between plot samples
    t_next_plot = t + P.t_plot
    while t < t_next_plot: # updates control and dynamics at faster simulation rate
        r = reference.square(t)  # reference input
        d = disturbance.step(t)  # input disturbance
        n = 0.0  #noise.random(t)  # simulate sensor noise
        x = ballbeam.state
        u = controller.update(r, x)  # update controller
        y = ballbeam.update(u + d)  # propagate system
        t = t + P.Ts  # advance time by Ts
    # update animation and data plots
    animation.update(ballbeam.state)
    dataPlot.update(t, r, ballbeam.state, u)
    plt.pause(0.0001)  # the pause causes the figure to be displayed during the simulation

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
