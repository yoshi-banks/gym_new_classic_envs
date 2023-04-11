import matplotlib.pyplot as plt
import numpy as np
import gym_new_classic_envs.envs.planar_vtol.vtol_resources.VTOLParam as P
from gym_new_classic_envs.utils.signalGenerator import signalGenerator
from gym_new_classic_envs.envs.planar_vtol.vtol_resources.VTOLAnimation import VTOLAnimation
from gym_new_classic_envs.envs.planar_vtol.vtol_resources.VTOLDataPlotter import dataPlotter

# instantiate reference input classes
reference = signalGenerator(amplitude=0.5, frequency=0.1)
zRef = signalGenerator(amplitude=5, frequency=0.01)
thetaRef = signalGenerator(amplitude=.125*np.pi, frequency=.5)
hRef = signalGenerator(amplitude=2, frequency=.2, y_offset=5)

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = VTOLAnimation()

t = P.t_start  # time starts at t_start
while t < P.t_end:  # main simulation loop
    # set variables
    r = reference.square(t)
    z = zRef.sin(t)
    theta = thetaRef.random(t)
    h = hRef.sin(t)
    # update animation
    state = np.array([[z], [h], [theta], [0.0], [0.0]])
    animation.update(state, z)
    dataPlot.update(t, state, z, h, [0.0], [0.0])

    t = t + P.t_plot  # advance time by t_plot
    plt.pause(0.05)

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
