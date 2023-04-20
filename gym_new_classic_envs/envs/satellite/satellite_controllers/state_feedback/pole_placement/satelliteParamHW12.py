# satellite Parameter File
import numpy as np
import control as cnt
import sys
sys.path.append('..')  # add parent directory
import gym_new_classic_envs.envs.satellite.satellite_resources.satelliteParam as P

# import variables from satelliteParam 
Ts = P.Ts
sigma = P.sigma
beta = P.beta
tau_max = P.tau_max

# tuning parameters
wn_th = 0.6
wn_phi = 1.1    # rise time for angle
zeta_phi = 0.707  # damping ratio position
zeta_th = 0.707  # damping ratio angle
integrator_pole = np.array([-1])


# State Space Equations
# xdot = A*x + B*u
# y = C*x
A = np.array([[0.0, 0.0,               1.0,      0.0],
               [0.0, 0.0,               0.0,      1.0],
               [-P.k/P.Js, P.k/P.Js, -P.b/P.Js, P.b/P.Js],
               [P.k/P.Jp, -P.k/P.Jp, P.b/P.Jp, -P.b/P.Jp]])

B = np.array([[0.0],
               [0.0],
               [1.0/P.Js],
               [0.0]])

C = np.array([[1.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0]])

# form augmented system
Cout = np.array([[0.0, 1.0, 0.0, 0.0]])
A1 = np.array([[0.0, 0.0, 1.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 1.0, 0.0],
               [-P.k/P.Js, P.k/P.Js, -P.b/P.Js, P.b/P.Js, 0.0],
               [P.k/P.Jp, -P.k/P.Jp, P.b/P.Jp, -P.b/P.Jp, 0.0],
               [0.0, -1.0, 0.0, 0.0, 0.0]])

B1 = np.array([[0.0],
               [0.0],
               [1.0/P.Js],
               [0.0],
               [0.0]])


# gain calculation
des_char_poly = np.convolve(
    np.convolve([1, 2*zeta_phi*wn_phi, wn_phi**2],
                [1, 2*zeta_th*wn_th, wn_th**2]),
    np.poly(integrator_pole))
des_poles = np.roots(des_char_poly)

# Compute the gains if the system is controllable
if np.linalg.matrix_rank(cnt.ctrb(A1, B1)) != 5:
    print("The system is not controllable")
else:
    K1 = cnt.place(A1, B1, des_poles)
    K = np.array([K1.item(0), K1.item(1), K1.item(2), K1.item(3)])
    ki = K1.item(4)

print('K: ', K)
print('ki: ', ki)



