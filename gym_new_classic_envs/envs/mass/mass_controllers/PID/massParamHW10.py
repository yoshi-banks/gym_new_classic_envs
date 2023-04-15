# Single link arm Parameter File
import numpy as np
# import control as cnt
import sys
sys.path.append('..')
import gym_new_classic_envs.envs.mass.mass_resources.massParam as P

Ts = P.Ts # sample rate of the controller
beta = P.beta # dirty derivative gain
P.F_max = P.F_max # limit on control signal

# tuning parameters
tr = 1.0
zeta = 0.7

# compute PD gains
# open loop char polynomial and poles
a1 = P.b/P.m
a0 = P.k/P.m

wn = 2.2/tr
alpha1 = 2.0*zeta*wn
alpha0 = wn**2

kp = P.m*(alpha0-a0)
kd = P.m*(alpha1-a1)
ki = 5.0

# using the values obtained from the PDF solution file _.
# kp = 6.0
# kd = 8.89

# kp = 3.050
# ki = 1.5
# kd = 7.277

print('kp: ', kp)
print('kd: ', kd)
print('ki: ', ki)