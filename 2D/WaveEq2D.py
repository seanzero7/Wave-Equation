import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

###############################################
#   WAVE EQUATION SIMULATION (2D)
#
#   Equation:
#   u_tt = c^2 (u_xx + u_yy)
#
#   Discretization (Finite Difference Method):
#   Using second-order differences in both x and y:
#
#   u_{i,j}^{n+1} = 2*u_{i,j}^n - u_{i,j}^{n-1} 
#                   + (c^2 * Δt^2 / Δx^2) * (u_{i+1,j}^n - 2*u_{i,j}^n + u_{i-1,j}^n)
#                   + (c^2 * Δt^2 / Δy^2) * (u_{i,j+1}^n - 2*u_{i,j}^n + u_{i,j-1}^n)
#
#   Boundary conditions: u=0 at all boundaries.
#   Initial condition: a Gaussian pulse in the center of the domain.
###############################################

#Parameters
L = 1.0            #Length of the domain in both x and y
nx = 100            #Number of x points
ny = 100            #Number of y points
c = 1.0             #Wave speed
dx = L/(nx-1)
dy = L/(ny-1)
dt = 0.5*min(dx,dy)/c  #Stable time step based on CFL

x = np.linspace(0, L, nx)
y = np.linspace(0, L, ny)
X, Y = np.meshgrid(x, y)

#Gaussian Pulse in the center
u0 = np.exp(-((X - 0.5*L)**2 + (Y - 0.5*L)**2)/0.01)
u_prev = np.copy(u0)
u_current = np.copy(u0)

#Dirichlet
u_prev[0,:] = 0.0; u_prev[-1,:] = 0.0; u_prev[:,0] = 0.0; u_prev[:,-1] = 0.0
u_current[0,:] = 0.0; u_current[-1,:] = 0.0; u_current[:,0] = 0.0; u_current[:,-1] = 0.0

def wave_step(u_current, u_prev, c, dx, dy, dt):
    u_next = np.zeros_like(u_current)
    #Computes second derivatives in x and y
    u_next[1:-1,1:-1] = (2*u_current[1:-1,1:-1] - u_prev[1:-1,1:-1]
                         + (c**2 * dt**2 / dx**2)*(u_current[2:,1:-1] - 2*u_current[1:-1,1:-1] + u_current[:-2,1:-1])
                         + (c**2 * dt**2 / dy**2)*(u_current[1:-1,2:] - 2*u_current[1:-1,1:-1] + u_current[1:-1,:-2]))
    #Dirichlet
    u_next[0,:] = 0.0
    u_next[-1,:] = 0.0
    u_next[:,0] = 0.0
    u_next[:,-1] = 0.0
    return u_next

fig_wave, ax_wave = plt.subplots()
im_wave = ax_wave.imshow(u_current, origin='lower', extent=[0,L,0,L], vmin=-1, vmax=1, cmap='RdBu')
ax_wave.set_title('2D Wave Equation Simulation')
ax_wave.set_xlabel('x')
ax_wave.set_ylabel('y')
fig_wave.colorbar(im_wave, ax=ax_wave, label='Displacement')

def update_wave(frame):
    global u_current, u_prev
    u_next = wave_step(u_current, u_prev, c, dx, dy, dt)
    u_prev = u_current
    u_current = u_next
    im_wave.set_data(u_current)
    return [im_wave]

###############################################
#   MAXWELL'S EQUATIONS SIMULATION (2D)
#
#   Consider a 2D electromagnetic wave with fields Ez, Hx, Hy.
#
#   In free space (no charges, no currents), using the Yee scheme:
#
#   Ez^{n+1}(i,j) = Ez^n(i,j) + (c * dt/dx)*(Hy(i,j)-Hy(i-1,j)) - (c * dt/dy)*(Hx(i,j)-Hx(i,j-1))
#   Hx^{n+1}(i,j) = Hx^n(i,j) - (c * dt/dy)*(Ez(i,j)-Ez(i,j-1))
#   Hy^{n+1}(i,j) = Hy^n(i,j) + (c * dt/dx)*(Ez(i-1,j)-Ez(i,j))
#
#   We'll assume a Gaussian pulse in Ez at t=0.
#   Boundaries: Ez=0 at the domain edges for simplicity.
###############################################

L_em = 1.0
nx_em = 100
ny_em = 100
c_em = 1.0
dx_em = L_em/(nx_em-1)
dy_em = L_em/(ny_em-1)
dt_em = 0.5*min(dx_em,dy_em)/c_em

x_em = np.linspace(0,L_em,nx_em)
y_em = np.linspace(0,L_em,ny_em)
X_em, Y_em = np.meshgrid(x_em, y_em)

#Fields
Ez = np.exp(-((X_em-0.5*L_em)**2 + (Y_em-0.5*L_em)**2)/0.01)
Hx = np.zeros((ny_em, nx_em))
Hy = np.zeros((ny_em, nx_em))

def maxwell_step(Ez, Hx, Hy, c, dx, dy, dt):
    #Updates Ez
    Ez_new = np.copy(Ez)
    Ez_new[1:-1,1:-1] = (Ez[1:-1,1:-1] 
                         + (c*dt/dx)*(Hy[1:-1,1:-1]-Hy[1:-1,0:-2]) 
                         - (c*dt/dy)*(Hx[1:-1,1:-1]-Hx[0:-2,1:-1]))
    #Boundaries Ez=0
    Ez_new[0,:] = 0.0; Ez_new[-1,:] = 0.0; Ez_new[:,0] = 0.0; Ez_new[:,-1] = 0.0

    #Updates Hx
    Hx_new = np.copy(Hx)
    Hx_new[1:-1,1:-1] = Hx[1:-1,1:-1] - (c*dt/dy)*(Ez[1:-1,1:-1]-Ez[0:-2,1:-1])

    #Updates Hy
    Hy_new = np.copy(Hy)
    Hy_new[1:-1,1:-1] = Hy[1:-1,1:-1] + (c*dt/dx)*(Ez[1:-1,1:-1] - Ez[1:-1,0:-2])

    return Ez_new, Hx_new, Hy_new

fig_em, ax_em = plt.subplots()
im_em = ax_em.imshow(Ez, origin='lower', extent=[0,L_em,0,L_em], vmin=-1, vmax=1, cmap='RdBu')
ax_em.set_title("2D Maxwell's Equations Simulation")
ax_em.set_xlabel('x')
ax_em.set_ylabel('y')
fig_em.colorbar(im_em, ax=ax_em, label='Ez Field Amplitude')

def update_em(frame):
    global Ez, Hx, Hy
    Ez, Hx, Hy = maxwell_step(Ez, Hx, Hy, c_em, dx_em, dy_em, dt_em)
    im_em.set_data(Ez)
    return [im_em]

###############################################
# Create the animations
###############################################
ani_wave = FuncAnimation(fig_wave, update_wave, frames=900, interval=30, blit=True)
ani_em = FuncAnimation(fig_em, update_em, frames=300, interval=30, blit=True)
ani_wave.save('2DWave_equation.gif', writer='imagemagick', fps=20)
ani_em.save('2DMaxwell_equation.gif', writer='imagemagick', fps=10)
plt.show()
