"""
Complete VTOL Simulation with Full Dynamics and Passivity-Based Control
Following "Modeling and Passivity-Based Control for a Convertible Fixed-Wing VTOL"
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters (from article Table 1)
m = 3.1
g = 9.81
rho = 1.2041
S = 0.75
b = 2.1
c = 0.3572
Ixx = 1.229
Iyy = 0.1702
Izz = 0.8808
Ixz = 0.9343
L1 = 0.57
L2 = 0.325
L3 = 0.129
KF = 1.97e-6
KM = 2.88e-7

# Aerodynamic coefficients
CD0, CDa, CDq = 0.0101, 0.8461, 0.0
CL0, CLa, CLq = 0.0254, 4.0191, 3.8954
CN0, CNb, CNp, CNr = 3.205e-18, -0.195, -0.117, 0.096

# Control gains
Kp_pos = np.diag([23, 28, 23])
Kp_att = np.diag([46, 57, 46])
Kv = np.diag([10, 15, 10, 18, 22, 18])

# ... imports e parâmetros ...

def desired_attitude(t):
    ## Primeira trajetória: linha reta
    theta = 0
    phi = 0
    psi = 0
    return theta, phi, psi

def desired_trajectory(t):
    # Segunda trajetória: linha costura
    if t < 3:
        pos = np.array([0, 0, -5])
    elif 3 <= t < 15:
        pos = np.array([2 * t, np.sin(2 * t), -5])
    elif 15 <= t <= 20:
        pos = np.array([2 * t, 0, 0])
    else:
        pos = np.array([40, 0, 0])
    
    theta, phi, psi = desired_attitude(t)
    return np.concatenate([pos, [theta, phi, psi]])

# Mixing matrix from the article
Y = np.array([
    [0, 0, KF, 0, KF, 0, KF],
    [0, -KF, 0, 0, 0, 0, 0],
    [-KF, 0, 0, -KF, 0, -KF, 0],
    [0, 0, 0, L3*KF, 0, -L3*KF, 0],
    [-L1*KF, 0, 0, L2*KF, 0, L2*KF, 0],
    [KM, -L1*KF, 0, KM, L3*KF, KM, -L3*KF]
])

# Dynamics function
def dynamics(t, state):
    pos = state[0:3]
    vel = state[3:6]
    att = state[6:9]
    omega = state[9:12]

    lam = np.concatenate([pos, att])
    dlam = np.concatenate([vel, omega])
    lam_des = desired_trajectory(t)

    e = lam - lam_des
    edot = dlam

    # Passivity-based control law
    delta_es = -np.concatenate([Kp_pos @ e[0:3], Kp_att @ e[3:6]])
    delta_di = -Kv @ edot
    delta = delta_es + delta_di

    # Control allocation (pseudo-inverse)
    u = np.linalg.pinv(Y) @ delta

    # Motor decomposition to forces
    Fx = Y[0] @ u + m * g * np.sin(att[0])
    Fy = Y[1] @ u - m * g * np.cos(att[0]) * np.sin(att[1])
    Fz = Y[2] @ u - m * g * np.cos(att[0]) * np.cos(att[1])
    Mx = Y[3] @ u
    My = Y[4] @ u
    Mz = Y[5] @ u

    acc = np.array([Fx, Fy, Fz]) / m
    I = np.array([[Ixx, 0, Ixz],                                
              [0, Iyy, 0],
              [Ixz, 0, Izz]])
    ang_acc = np.linalg.inv(I) @ np.array([Mx, My, Mz])
    # ang_acc = np.linalg.inv(np.diag([Ixx, Iyy, Izz])) @ np.array([Mx, My, Mz])

    dstate = np.zeros(12)
    dstate[0:3] = vel
    dstate[3:6] = acc
    dstate[6:9] = omega
    dstate[9:12] = ang_acc

    return dstate

# Initial state
state0 = np.zeros(12)

# Time span
t_span = (0, 60)
t_eval = np.linspace(0, 60, 3000)

# Solve ODE
sol = solve_ivp(dynamics, t_span, state0, t_eval=t_eval, rtol=1e-6)



# Plot results com referência pontilhada
plt.figure(figsize=(10, 6))
# Trajetória simulada
plt.plot(sol.t, sol.y[0], label='x (m)')
plt.plot(sol.t, sol.y[1], label='y (m)')
plt.plot(sol.t, sol.y[2], label='z (m)')

# Trajetória de referência (pontilhada)
ref_traj = np.array([desired_trajectory(t) for t in sol.t])
plt.plot(sol.t, ref_traj[:, 0], '--', label='x ref (m)', color='tab:blue', alpha=0.5)
plt.plot(sol.t, ref_traj[:, 1], '--', label='y ref (m)', color='tab:orange', alpha=0.5)
plt.plot(sol.t, ref_traj[:, 2], '--', label='z ref (m)', color='tab:green', alpha=0.5)

plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('VTOL Spiral Trajectory Tracking')
plt.legend()
plt.grid()
# plt.show()







# Simulated attitude angles
theta = sol.y[6]
phi = sol.y[7]
psi = sol.y[8]
# Reference attitude angles
ref_angles = np.array([desired_attitude(t) for t in sol.t])
theta_ref = ref_angles[:, 0]
phi_ref   = ref_angles[:, 1]
psi_ref   = ref_angles[:, 2]

# Plot attitude angles with reference (without control as dashed)
plt.figure(figsize=(10, 8))
# Phi plot
plt.subplot(3, 1, 2)
plt.plot(sol.t, phi, label='Phi (controlled)', color='g')
plt.plot(sol.t, phi_ref, '--', label='Phi (reference)', color='gray')
plt.title('Phi (Roll)')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.ylim(-5e-3, 10e-3) 
plt.ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))
plt.grid()
plt.legend()
# Theta plot
plt.subplot(3, 1, 1)
plt.plot(sol.t, theta, label='Theta (controlled)', color='b')
plt.plot(sol.t, theta_ref, '--', label='Theta (reference)', color='gray')
plt.title('Theta (Pitch)')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.ylim(-10, 10)
plt.grid()
plt.legend()
# Psi plot
plt.subplot(3, 1, 3)
plt.plot(sol.t, psi, label='Psi (controlled)', color='r')
plt.plot(sol.t, psi_ref, '--', label='Psi (reference)', color='gray')
plt.title('Psi (Yaw)')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.grid()
plt.legend()

plt.tight_layout()
# plt.show()







# 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(sol.y[0], sol.y[1], sol.y[2], label='Trajectory')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Trajectory of VTOL')
ax.legend()
plt.show()
