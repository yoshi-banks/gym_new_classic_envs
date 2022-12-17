import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))  # add parent directory
from gym_new_classic_envs.envs.satellite.satellite_resources import satelliteParam as P
from gym_new_classic_envs.utils.signalGenerator import signalGenerator
from gym_new_classic_envs.envs.satellite.satellite_resources.satelliteAnimation import satelliteAnimation
from gym_new_classic_envs.envs.satellite.satellite_resources.satelliteDataPlotter import dataPlotter


# instantiate reference input classes
reference = signalGenerator(amplitude=0.5, frequency=0.1)
thetaRef = signalGenerator(amplitude=2.0*np.pi, frequency=0.1)
phiRef = signalGenerator(amplitude=0.5, frequency=0.1)
tauRef = signalGenerator(amplitude=5, frequency=.5)

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = satelliteAnimation()

t = P.t_start  # time starts at t_start
while t < P.t_end:  # main simulation loop

    # set variables
    r = reference.square(t)
    theta = thetaRef.sin(t)
    phi = phiRef.sin(t)
    tau = tauRef.sawtooth(t)

    # update animation
    state = np.array([[theta], [phi], [0.0], [0.0]])
    animation.update(state)
    dataPlot.update(t, r, state, tau)

    # advance time by t_plot
    t = t + P.t_plot  
    plt.pause(0.1)

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
