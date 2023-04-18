# satellite Parameter File
import numpy as np
import control as cnt
import sys
sys.path.append('..')  # add parent directory
import os
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))  # add parent directory
import gym_new_classic_envs.envs.planar_vtol.vtol_resources.VTOLParam as P

# import variables from satelliteParam
Ts = P.Ts
sigma = P.sigma
beta = P.beta
F_max = P.F_max
tau_max = P.tau_max
Fe = P.Fe

# tuning parameters
tr_h = 2.0
zeta_h = 0.707
tr_z = 2.0
M = 10.0
zeta_z = 0.707
zeta_th = 0.707
tr_th = tr_z/M

wn_th = 2.2/tr_th
wn_h = 2.2/tr_h
wn_z = 2.2/tr_z

# State Space Equations
# xdot = A*x + B*u
# y = C*x
A_lon = np.array([[0.0, 1.0],
                  [0.0, 0.0]])
B_lon = np.array([[0.0],
                  [1.0/(P.mc+2*P.mr)]])
C_lon = np.array([[1.0, 0.0]])
D_lon = np.array([0.0])

A_lat = np.array([[0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0],
              [0.0, (-P.Fe/(P.mc+2*P.mr)), (-P.mu/(P.mc+2*P.mr)), 0.0],
              [0.0, 0.0, 0.0, 0.0]])
B_lat = np.array([[0.0],
               [0.0],
               [0.0],
               [(1.0/(P.Jc+2*P.mr*P.d**2))]])
C_lat = np.array([[1.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0]])

D_lat = np.array([[0.0],
              [0.0]])

# gain calculation
des_char_poly_z = np.convolve([1, 2*zeta_th*wn_th, wn_th**2],
                            [1, 2*zeta_z*wn_z, wn_z**2])
des_poles_z = np.roots(des_char_poly_z)

des_char_poly_h = [1, 2*zeta_h*wn_h, wn_h**2]
des_poles_h = np.roots(des_char_poly_h)

# Compute the gains if the system is controllable
if np.linalg.matrix_rank(cnt.ctrb(A_lon, B_lon)) != 2:
    print("The longitudinal system is not controllable")
else:
    K_lon = cnt.acker(A_lon, B_lon, des_poles_h)
    Cr_lon = np.array([[1.0, 0.0]])
    kr_lon = -1.0/(Cr_lon @ np.linalg.inv(A_lon - B_lon @ K_lon) @ B_lon)

if np.linalg.matrix_rank(cnt.ctrb(A_lat, B_lat)) != 4:
    print("The lateral system is not controllable")
else:
    K_lat = cnt.acker(A_lat, B_lat, des_poles_z)
    Cr_lat = np.array([[1.0, 0.0, 0.0, 0.0]])
    kr_lat = -1.0/(Cr_lat @ np.linalg.inv(A_lat - B_lat @ K_lat) @ B_lat)


print('K_lon: ', K_lon)
print('kr_lon: ', kr_lon)
print('K_lat: ', K_lat)
print('kr_lat: ', kr_lat)



