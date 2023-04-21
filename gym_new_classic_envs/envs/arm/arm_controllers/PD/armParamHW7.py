# Single link arm Parameter File
import numpy as np
# import control as cnt
import sys
sys.path.append('..')  # add parent directory
import gym_new_classic_envs.envs.arm.arm_resources.armParam as P

Ts = P.Ts  # sample rate of the controller
beta = P.beta  # dirty derivative gain
tau_max = P.tau_max  # limit on control signal

# PD gains
kp = 0.18
kd = 0.095

print('kp: ', kp)
print('kd: ', kd)



