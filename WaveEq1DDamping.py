import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

###############################################
#   WAVE EQUATION SIMULATION (1D) with Damping
#
#   Damped Wave Equation:
#   u_tt = c^2 u_xx - α u_t
#
#   Discretization leads to:
#   u_i^{n+1} = 2*u_i^n - u_i^{n-1} 
#                + (c^2 * Δt^2 / Δx^2) * (u_{i+1}^n - 2*u_i^n + u_{i-1}^n)
#                - α Δt (u_i^n - u_i^{n-1})
#
###############################################

#Parameters
L = 1.0         #Length
nx = 200         #Number of spatial points
c = 1.0          #Wave speed
dx = L/(nx-1)
dt = 0.5*dx/c    #Stable time step based on CFL condition
alpha = 0.2      #Damping coefficient

x = np.linspace(0, L, nx)

#Gaussian pulse
u0 = np.exp(-((x - 0.5*L)**2)/(0.01))

#Initial velocity = 0
u_prev = np.copy(u0)
u_current = np.copy(u0)

#Dirichlet
u_prev[0] = 0.0
u_prev[-1] = 0.0
u_current[0] = 0.0
u_current[-1] = 0.0

def wave_step(u_current, u_prev, c, dx, dt, alpha):
    u_next = np.zeros_like(u_current)
    u_next[1:-1] = (2*u_current[1:-1] - u_prev[1:-1]
                    + (c**2 * dt**2 / dx**2)*(u_current[2:] - 2*u_current[1:-1] + u_current[:-2])
                    - alpha*dt*(u_current[1:-1] - u_prev[1:-1]))
    u_next[0] = 0.0
    u_next[-1] = 0.0
    return u_next

#Wave equation animation
fig_wave, ax_wave = plt.subplots()
line_wave, = ax_wave.plot(x, u_current, color='blue', lw=2)
ax_wave.set_ylim(-1.0, 1.0)
ax_wave.set_xlim(0, L)
ax_wave.set_title('Damped Wave Equation Simulation')
ax_wave.set_xlabel('Position')
ax_wave.set_ylabel('Displacement')

def update_wave(frame):
    global u_current, u_prev
    u_next = wave_step(u_current, u_prev, c, dx, dt, alpha)
    u_prev, u_current = u_current, u_next
    line_wave.set_ydata(u_current)
    return line_wave,


###############################################
#   MAXWELL'S EQUATIONS SIMULATION (1D) with Damping
#
#   In free space:
#   dE/dt = c^2 dH/dx
#   dH/dt = c^2 dE/dx
#
#   Discretized (Yee):
#   E^{n+1}_i = E^n_i + (c Δt/Δx) * (H^n_{i-1} - H^n_{i})
#   H^{n+1}_i = H^n_i + (c Δt/Δx) * (E^{n+1}_i - E^{n+1}_{i-1})
#
#   With damping:
#   After updating E and H as above, apply:
#   E^{n+1}_i = (1 - β Δt) E^{n+1}_i
#   H^{n+1}_i = (1 - β Δt) H^{n+1}_i
#
###############################################

# Parameters for Maxwell equations
L_em = 1.0
nx_em = 200
c_em = 1.0
dx_em = L_em/(nx_em-1)
dt_em = 0.5*dx_em/c_em
beta = 0.2  #Damping coefficient for EM fields

x_em = np.linspace(0, L_em, nx_em)

#Initial fields
E = np.exp(-((x_em - 0.5*L_em)**2)/(0.01))
H = np.zeros_like(E)

def maxwell_step(E, H, c, dx, dt, beta):
    E_new = np.copy(E)
    H_new = np.copy(H)

    #Updates E
    E_new[1:] = E[1:] + (c*dt/dx)*(H[:-1] - H[1:])
    E_new[0] = 0.0
    E_new[-1] = 0.0

    #Updates H
    H_new[:-1] = H[:-1] + (c*dt/dx)*(E_new[:-1] - E_new[1:])

    #Damping
    damping_factor = (1 - beta*dt)
    if damping_factor < 0:
        damping_factor = 0.0  #Ensures no negative scaling
    E_new *= damping_factor
    H_new *= damping_factor

    return E_new, H_new

fig_em, ax_em = plt.subplots()
line_E, = ax_em.plot(x_em, E, label='E-field', color='red')
line_H, = ax_em.plot(x_em, H, label='H-field', color='green')
ax_em.set_ylim(-1.0, 1.0)
ax_em.set_xlim(0, L_em)
ax_em.set_title("Damped 1D Maxwell's Equations Simulation")
ax_em.set_xlabel('Position')
ax_em.set_ylabel('Field Amplitude')
ax_em.legend()

def update_em(frame):
    global E, H
    E, H = maxwell_step(E, H, c_em, dx_em, dt_em, beta)
    line_E.set_ydata(E)
    line_H.set_ydata(H)
    return line_E, line_H

###############################################
# Create the animations
###############################################

ani_wave = FuncAnimation(fig_wave, update_wave, frames=200, interval=30, blit=True)
ani_em = FuncAnimation(fig_em, update_em, frames=200, interval=30, blit=True)

plt.show()
