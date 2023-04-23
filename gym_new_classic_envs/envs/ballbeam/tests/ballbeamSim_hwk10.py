import matplotlib.pyplot as plt
import gym_new_classic_envs.envs.ballbeam.ballbeam_resources.ballbeamParam as P
from gym_new_classic_envs.utils.signalGenerator import signalGenerator
from gym_new_classic_envs.envs.ballbeam.ballbeam_resources.ballbeamAnimation import ballbeamAnimation
from gym_new_classic_envs.envs.ballbeam.ballbeam_resources.ballbeamDataPlotter import dataPlotter
from gym_new_classic_envs.envs.ballbeam.ballbeam_resources.ballbeamDynamics import ballbeamDynamics
from gym_new_classic_envs.envs.ballbeam.ballbeam_controllers.PID.ballbeamController import ballbeamController

# instantiate satellite, controller, and reference classes
ballbeam = ballbeamDynamics(alpha=0.20)
controller = ballbeamController()
reference = signalGenerator(amplitude=0.125, frequency=0.02, y_offset=0.0)
disturbance = signalGenerator(amplitude=0.25, frequency=0.0)
noise = signalGenerator(amplitude=0.01)

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = ballbeamAnimation()

t = P.t_start  # time starts at t_start
y = ballbeam.h()
while t < P.t_end:  # main simulation loop

    # Propagate dynamics in between plot samples
    t_next_plot = t + P.t_plot

    # updates control and dynamics at faster simulation rate
    while t < t_next_plot:
        r = reference.step(t)
        d = 0.0 #disturbance.step(t)
        n = 0.0 #noise.random(t)
        # x = ballbeam.state
        # u = controller.update(r, x)
        u = controller.update(r, y + n)
        y = ballbeam.update(u + d)  # Propagate the dynamics
        t = t + P.Ts  # advance time by Ts

    # update animation and data plots
    animation.update(ballbeam.state)
    dataPlot.update(t, r, ballbeam.state, u)

    # the pause causes the figure to be displayed during the
    # simulation
    plt.pause(0.0001)

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
