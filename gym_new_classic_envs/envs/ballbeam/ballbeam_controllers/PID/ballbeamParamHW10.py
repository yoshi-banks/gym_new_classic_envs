import numpy as np

import gym_new_classic_envs.envs.ballbeam.ballbeam_resources.ballbeamParam as P

##########################################3
#           PD control: Time design strategy
############################################
# tuning parameters
tr_z = 1.2 # rise time for outer loop
# tr_z = 1.0 # tuned for fastest rise time without saturation
zeta_z = 0.707 # damping ratio for outer loo
M = 10.0 # time scale separation between inner and outer loop
zeta_th = 0.707 # damping ratio for inner loop
ki_z = -0.1

# saturation limits
F_max = 20.0
theta_max = 30.0*np.pi/180.0

# PD design for inner loop
ze = P.length/2.0 # equilibrium position - center of beam
b0 = P.length/((P.m2*P.length**2/3.0)+(P.m1*ze**2))
tr_theta = tr_z/M # rise time for inner loop
wn_th = 2.2/tr_theta # natural frequency for inner loop
kp_th = wn_th**2/b0 # kp - inner
kd_th = 2.0*zeta_th*wn_th/b0 # kd - inner

# DC gain for inner loop
k_DC_th = 1.0

# PD design for outer loop
wn_z = 2.2/tr_z # natural frequency - outer loop
kp_z = -wn_z**2/P.g/k_DC_th # kp - outer
kd_z = -2.0*zeta_z*wn_z/P.g/k_DC_th # kd - outer

# Austin's controller coefficients
kp_th = 126.74189814814818
kd_th = 9.7752569444444445
kp_z = -0.034262090837014386
kd_z = -0.2642541625193
ki_z = -0.1

print('DC gain', k_DC_th)
print('kp_th', kp_th)
print('kd_th', kd_th)
print('kp_z', kp_z)
print('kd_z', kd_z)

