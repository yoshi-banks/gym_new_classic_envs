import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))  # add parent directory
import gym_new_classic_envs.envs.ballbeam.ballbeam_resources.ballbeamParam as P
from gym_new_classic_envs.utils.signalGenerator import signalGenerator
from gym_new_classic_envs.envs.ballbeam.ballbeam_resources.ballbeamAnimation import ballbeamAnimation
from gym_new_classic_envs.envs.ballbeam.ballbeam_resources.ballbeamDataPlotter import dataPlotter

# instantiate reference input classes
reference = signalGenerator(amplitude=0.5, frequency=0.1)
zRef = signalGenerator(amplitude=0.25, frequency=0.1)
thetaRef = signalGenerator(amplitude=.125*np.pi, frequency=.1, y_offset=.25*np.pi)
fRef = signalGenerator(amplitude=5, frequency=.5)

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = ballbeamAnimation()

t = P.t_start  # time starts at t_start
while t < P.t_end:  # main simulation loop
    # set variables
    r = reference.square(t)
    z = zRef.sin(t)
    theta = thetaRef.sin(t)
    f = fRef.sawtooth(t)
    # update animation
    state = np.array([[z], [theta], [0.0], [0.0]])
    animation.update(state)
    dataPlot.update(t, r, state, f)

    t = t + P.t_plot  # advance time by t_plot
    plt.pause(0.05)

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
