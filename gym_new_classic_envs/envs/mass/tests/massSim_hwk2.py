import matplotlib.pyplot as plt
import numpy as np
import sys
# sys.path.append('..')  # add parent directory
import gym_new_classic_envs.envs.mass.mass_resources.massParam as P
from gym_new_classic_envs.envs.mass.mass_resources.signalGenerator import signalGenerator
from gym_new_classic_envs.envs.mass.mass_resources.massAnimation import massAnimation
from gym_new_classic_envs.envs.mass.mass_resources.dataPlotter import dataPlotter


# instantiate reference input classes
reference = signalGenerator(amplitude=.5, frequency=0.1)
thetaRef = signalGenerator(amplitude=2.0*np.pi, frequency=0.1)
tauRef = signalGenerator(amplitude=5, frequency=.5)

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = massAnimation()

t = P.t_start  # time starts at t_start
while t < P.t_end:  # main simulation loop
    # set variables
    r = reference.square(t)
    theta = thetaRef.sin(t)
    tau = tauRef.sawtooth(t)
    # update animation

    state = np.array([[theta], [0.0]])
    animation.update(state)
    dataPlot.update(t, r, state, tau)

    #plt.show()
    t = t + P.t_plot  # advance time by t_plot
    plt.pause(0.1)

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()


