# Inverted Pendulum Parameter File
import numpy as np
import control as cnt
import gym_new_classic_envs.envs.ballbeam.ballbeam_resources.ballbeamParam as P

ze = P.ze
m1 = P.m1
g = P.g
length = P.length
m2 = P.m2

# sample rate of the controller
Ts = P.Ts

# saturation limits
F_max = P.F_max                # Max Force, N
theta_max = 30.0*np.pi/180.0  # Max theta, rads

####################################################
#                 State Space
####################################################
# tuning parameters
tr_z = 1.5             # rise time for position
tr_theta = 0.25         # rise time for angle
zeta_z   = 0.707       # damping ratio position
zeta_th  = 0.707       # damping ratio angle
integrator_pole = np.array([-0.5]) # integrator_pole = [-2]  # integrator pole

# State Space Equations
# xdot = A*x + B*u
# y = C*x
A = np.array([[0.0, 0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0, 1.0],
               [0.0, -P.g, 0.0, 0.0],
               [-(P.m1*P.g)/((P.m2*P.length**2)/3.0+
                             P.m1*(P.length/2.0)**2),
                0.0, 0.0, 0.0]])

B = np.array([[0.0],
               [0.0],
               [0.0],
               [P.length/(P.m2*P.length**2/3.0+
                          P.m1*P.length**2/4.0)]])

C = np.array([[1.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0]])

# form augmented system
Cout = np.array([[1.0, 0.0, 0.0, 0.0]])

A1 = np.array([[0.0, 0.0, 1.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 1.0, 0.0],
               [0.0, -P.g, 0.0, 0.0, 0.0],
               [-(P.m1*P.g)/((P.m2*P.length**2)/3.0+
                             P.m1*(P.length/2.0)**2),
                0.0, 0.0, 0.0, 0.0],
               [-1.0, 0.0, 0.0, 0.0, 0.0]])

B1 = np.array([[0.0],
               [0.0],
               [0.0],
               [P.length/(P.m2*P.length**2/3.0+
                          P.m1*(P.length/2.0)**2)],
               [0.0]])

Q = np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 10.0, 0.0, 0.0, 0.0],
               [0.0, 0, 1.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 1.0]])

R = np.array([[0.001]])

# gain calculation
wn_th = 2.2/tr_theta  # natural frequency for angle
wn_z = 2.2/tr_z  # natural frequency for position
des_char_poly = np.convolve(
    np.convolve([1, 2*zeta_z*wn_z, wn_z**2],
                [1, 2*zeta_th*wn_th, wn_th**2]),
    np.poly(integrator_pole))
des_poles = np.roots(des_char_poly)

# Compute the gains if the system is controllable
if np.linalg.matrix_rank(cnt.ctrb(A1, B1)) != 5:
    print("The system is not controllable")
else:
    # K1 = cnt.acker(A1, B1, des_poles)
    K1, S, E = cnt.lqr(A1, B1, Q, R)
    K = np.matrix([K1.item(0), K1.item(1),
                   K1.item(2), K1.item(3)])
    ki = K1.item(4)

print('K: ', K)
print('ki: ', ki)




