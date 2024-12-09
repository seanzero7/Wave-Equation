import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D 

###############################################
#   WAVE EQUATION SIMULATION (3D VISUALIZATION)
#
#   Equation:
#   u_tt = c^2 (u_xx + u_yy)
#
#   Discretization (Finite Difference Method):
#   u_{i,j}^{n+1} = 2*u_{i,j}^n - u_{i,j}^{n-1} 
#                   + (c^2 * Δt^2 / Δx^2) * (u_{i+1,j}^n - 2*u_{i,j}^n + u_{i-1,j}^n)
#                   + (c^2 * Δt^2 / Δy^2) * (u_{i,j+1}^n - 2*u_{i,j}^n + u_{i,j-1}^n)
#
#   Boundary conditions: u=0 at all boundaries.
#   Initial condition: a Gaussian pulse in the center of the domain.
###############################################

#Parameters
L = 1.0
nx = 100
ny = 100
c = 1.0
dx = L/(nx-1)
dy = L/(ny-1)
dt = 0.5*min(dx,dy)/c

x = np.linspace(0, L, nx)
y = np.linspace(0, L, ny)
X, Y = np.meshgrid(x, y)

u0 = np.exp(-((X - 0.5*L)**2 + (Y - 0.5*L)**2)/0.01)
u_prev = np.copy(u0)
u_current = np.copy(u0)

#Dirichlet
u_prev[0,:] = 0.0; u_prev[-1,:] = 0.0; u_prev[:,0] = 0.0; u_prev[:,-1] = 0.0
u_current[0,:] = 0.0; u_current[-1,:] = 0.0; u_current[:,0] = 0.0; u_current[:,-1] = 0.0

def wave_step(u_current, u_prev, c, dx, dy, dt):
    u_next = np.zeros_like(u_current)
    u_next[1:-1,1:-1] = (2*u_current[1:-1,1:-1] - u_prev[1:-1,1:-1]
                         + (c**2 * dt**2 / dx**2)*(u_current[2:,1:-1] - 2*u_current[1:-1,1:-1] + u_current[:-2,1:-1])
                         + (c**2 * dt**2 / dy**2)*(u_current[1:-1,2:] - 2*u_current[1:-1,1:-1] + u_current[1:-1,:-2]))
    #Dirichlet
    u_next[0,:] = 0.0
    u_next[-1,:] = 0.0
    u_next[:,0] = 0.0
    u_next[:,-1] = 0.0
    return u_next

#3D axis
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim(-1, 1)

#Initial surface plot
surf = [ax.plot_surface(X, Y, u_current, cmap='RdBu', vmin=-1, vmax=1, linewidth=0, antialiased=True)]

ax.set_title('3D Wave Equation Simulation')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u(x,y,t)')

def update(frame):
    global u_current, u_prev
    u_next = wave_step(u_current, u_prev, c, dx, dy, dt)
    u_prev = u_current
    u_current = u_next
    #Clear and replot surface
    ax.clear()
    ax.set_zlim(-1, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(x,y,t)')
    ax.set_title('3D Wave Equation Simulation')
    surf = ax.plot_surface(X, Y, u_current, cmap='RdBu', vmin=-1, vmax=1, linewidth=0, antialiased=True)
    return [surf]

ani = FuncAnimation(fig, update, frames=300, interval=30, blit=False)

ani.save('wave_equation.gif', writer='imagemagick', fps=30)

plt.show()