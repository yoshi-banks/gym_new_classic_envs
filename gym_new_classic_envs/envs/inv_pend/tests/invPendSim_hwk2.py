import matplotlib.pyplot as plt
import numpy as np
import sys
# sys.path.append('..')  # add parent directory
import gym_new_classic_envs.envs.inv_pend.inv_pend_resources.invPendParam as P
from gym_new_classic_envs.envs.inv_pend.inv_pend_resources.invPendDynamics import InvertedPendulumDynamics
from gym_new_classic_envs.utils.signalGenerator import signalGenerator
from gym_new_classic_envs.envs.inv_pend.inv_pend_resources.invPendAnimation import InvertedPendulumAnimation
from gym_new_classic_envs.utils.dataPlotter import dataPlotter

# instantiate reference input classes
reference = signalGenerator(amplitude=0.5, frequency=0.1)
zRef = signalGenerator(amplitude=0.5, frequency=0.1)
thetaRef = signalGenerator(amplitude=.25*np.pi, frequency=.5)
fRef = signalGenerator(amplitude=5, frequency=.5)

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = InvertedPendulumAnimation()

t = P.t_start  # time starts at t_start
while t < P.t_end:  # main simulation loop
    # set variables
    r = reference.square(t)
    z = zRef.sin(t)
    theta = thetaRef.square(t)
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
