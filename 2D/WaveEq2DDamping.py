import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

###############################################
#   DAMPED WAVE EQUATION SIMULATION (2D)
###############################################

#Parameters
L = 1.0            #Length of the domain in both x and y
nx = 100            #Number of x points
ny = 100            #Number of y points
c = 1.0             #Wave speed
dx = L/(nx-1)
dy = L/(ny-1)
dt = 0.5*min(dx,dy)/c  #Stable time step based on CFL
alpha = 0.2          #Damping coefficient

x = np.linspace(0, L, nx)
y = np.linspace(0, L, ny)
X, Y = np.meshgrid(x, y)

#Gaussian in the center
u0 = np.exp(-((X - 0.5*L)**2 + (Y - 0.5*L)**2)/0.01)
u_prev = np.copy(u0)
u_current = np.copy(u0)

#Dirichlet
u_prev[0,:] = 0.0; u_prev[-1,:] = 0.0; u_prev[:,0] = 0.0; u_prev[:,-1] = 0.0
u_current[0,:] = 0.0; u_current[-1,:] = 0.0; u_current[:,0] = 0.0; u_current[:,-1] = 0.0

def wave_step(u_current, u_prev, c, dx, dy, dt, alpha):
    u_next = np.zeros_like(u_current)
    u_next[1:-1,1:-1] = (2*u_current[1:-1,1:-1] - u_prev[1:-1,1:-1]
                         + (c**2 * dt**2 / dx**2)*(u_current[2:,1:-1] - 2*u_current[1:-1,1:-1] + u_current[:-2,1:-1])
                         + (c**2 * dt**2 / dy**2)*(u_current[1:-1,2:] - 2*u_current[1:-1,1:-1] + u_current[1:-1,:-2])
                         - alpha*dt*(u_current[1:-1,1:-1] - u_prev[1:-1,1:-1]))
    #Dirichlet
    u_next[0,:] = 0.0
    u_next[-1,:] = 0.0
    u_next[:,0] = 0.0
    u_next[:,-1] = 0.0
    return u_next

fig_wave, ax_wave = plt.subplots()
im_wave = ax_wave.imshow(u_current, origin='lower', extent=[0,L,0,L], vmin=-1, vmax=1, cmap='RdBu')
ax_wave.set_title('2D Damped Wave Equation Simulation')
ax_wave.set_xlabel('x')
ax_wave.set_ylabel('y')
fig_wave.colorbar(im_wave, ax=ax_wave, label='Displacement')

def update_wave(frame):
    global u_current, u_prev
    u_next = wave_step(u_current, u_prev, c, dx, dy, dt, alpha)
    u_prev, u_current = u_current, u_next
    im_wave.set_data(u_current)
    return [im_wave]

###############################################
#   DAMPED MAXWELL'S EQUATIONS SIMULATION (2D)
###############################################

L_em = 1.0
nx_em = 100
ny_em = 100
c_em = 1.0
dx_em = L_em/(nx_em-1)
dy_em = L_em/(ny_em-1)
dt_em = 0.5*min(dx_em,dy_em)/c_em
beta = 0.2  #Damping coefficient for EM fields

x_em = np.linspace(0,L_em,nx_em)
y_em = np.linspace(0,L_em,ny_em)
X_em, Y_em = np.meshgrid(x_em, y_em)

#Fields
Ez = np.exp(-((X_em-0.5*L_em)**2 + (Y_em-0.5*L_em)**2)/0.01)
Hx = np.zeros((ny_em, nx_em))
Hy = np.zeros((ny_em, nx_em))

def maxwell_step(Ez, Hx, Hy, c, dx, dy, dt, beta):
    Ez_new = np.copy(Ez)
    Hx_new = np.copy(Hx)
    Hy_new = np.copy(Hy)

    #Updates Ez
    Ez_new[1:-1,1:-1] = (Ez[1:-1,1:-1] 
                         + (c*dt/dx)*(Hy[1:-1,1:-1]-Hy[1:-1,0:-2]) 
                         - (c*dt/dy)*(Hx[1:-1,1:-1]-Hx[0:-2,1:-1]))
    #Boundaries Ez=0
    Ez_new[0,:] = 0.0; Ez_new[-1,:] = 0.0; Ez_new[:,0] = 0.0; Ez_new[:,-1] = 0.0

    #Updates Hx
    Hx_new[1:-1,1:-1] = Hx[1:-1,1:-1] - (c*dt/dy)*(Ez[1:-1,1:-1]-Ez[0:-2,1:-1])

    #Updates Hy
    Hy_new[1:-1,1:-1] = Hy[1:-1,1:-1] + (c*dt/dx)*(Ez[1:-1,1:-1] - Ez[1:-1,0:-2])

    # Damping fields
    damping_factor = (1 - beta*dt)
    if damping_factor < 0:
        damping_factor = 0.0
    Ez_new *= damping_factor
    Hx_new *= damping_factor
    Hy_new *= damping_factor

    return Ez_new, Hx_new, Hy_new

fig_em, ax_em = plt.subplots()
im_em = ax_em.imshow(Ez, origin='lower', extent=[0,L_em,0,L_em], vmin=-1, vmax=1, cmap='RdBu')
ax_em.set_title("2D Damped Maxwell's Equations Simulation")
ax_em.set_xlabel('x')
ax_em.set_ylabel('y')
fig_em.colorbar(im_em, ax=ax_em, label='Ez Field Amplitude')

def update_em(frame):
    global Ez, Hx, Hy
    Ez, Hx, Hy = maxwell_step(Ez, Hx, Hy, c_em, dx_em, dy_em, dt_em, beta)
    im_em.set_data(Ez)
    return [im_em]

###############################################
# Create the animations
###############################################
ani_wave = FuncAnimation(fig_wave, update_wave, frames=200, interval=30, blit=True)
ani_em = FuncAnimation(fig_em, update_em, frames=200, interval=30, blit=True)

plt.show()
