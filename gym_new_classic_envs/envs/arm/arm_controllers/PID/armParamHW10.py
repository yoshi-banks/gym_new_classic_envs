# Single link arm Parameter File
import numpy as np
# import control as cnt
import sys
sys.path.append('..')  # add parent directory
import gym_new_classic_envs.envs.arm.arm_resources.armParam as P

Ts = P.Ts  # sample rate of the controller
beta = P.beta  # dirty derivative gain
tau_max = P.tau_max  # limit on control signal

#  tuning parameters
#tr = 0.8 # part (a)
tr = 2.0  # tuned for fastest possible rise time before saturation. 0.6
zeta = 0.707
ki = 0.0025  # integrator gain

# desired natural frequency
# wn = 0.5*np.pi/(tr*np.sqrt(1-zeta**2))
wn = 2.2/tr
alpha1 = 2.0*zeta*wn
alpha0 = wn**2

# compute PD gains
kp = alpha0*(P.m*P.ell**2)/3.0
kd = (P.m*P.ell**2)/3.0*(alpha1-3.0*P.b/(P.m*P.ell**2))

print('kp: ', kp)
print('ki: ', ki)
print('kd: ', kd)



