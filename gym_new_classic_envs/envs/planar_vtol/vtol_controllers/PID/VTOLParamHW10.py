import numpy as np
import sys
sys.path.append('..')
import gym_new_classic_envs.envs.planar_vtol.vtol_resources.VTOLParam as P

# tuning parameters
tr_h = 2.0 # Rise time for altitude
zeta_h = 0.707 # Damping ratio for altitude
tr_z = 2.0 # rise time for outer lateral loop
M = 10.0 # separation parameter
zeta_z = 0.707 # damping ratio for outer lateral loop
zeta_th = 0.707 # damping ratio for inner lateral loop
max_thrust = P.F_max

# PD gains for logitudinal (altiutude) control
wn_h = 2.2/tr_h
Delta_cl_d = [1, 2*zeta_h*wn_h, wn_h**2.0] # desired closed loop char eq
kp_h = Delta_cl_d[2]*(P.mc+2.0*P.mr) # kp - altitude
kd_h = Delta_cl_d[1]*(P.mc+2.0*P.mr) # kd - altitude
Fe = (P.mc+2.0*P.mr)*P.g

# PID design for logitudinal loop
ki_h = 0.25

# PD gains for lateral inner loop
b0 = 1.0/(P.Jc+2.0*P.mr*P.d**2)
tr_th = tr_z/M
wn_th = 2.2/tr_th
kp_th = wn_th**2.0/b0
kd_th = 2.0*zeta_th*wn_th/b0

# PD gain for lateral outer loop
b1 = -P.Fe/(P.mc+2.0*P.mr)
a1 = P.mu/(P.mc+2.0*P.mr)
wn_z = 2.2/tr_z
kp_z = wn_z**2.0/b1
kd_z = (2.0*zeta_z*wn_z-a1)/b1

# PID design for outer loop
ki_z = -0.031
theta_max = 30.0*np.pi/180.0
tau_max = 0.5
F_max = P.F_max


print('kp_z: ', kp_z)
print('kd_z: ', kd_z)
print('kp_h: ', kp_h)
print('kd_h: ', kd_h)
print('kp_th: ', kp_th)
print('kd_th: ', kd_th)
