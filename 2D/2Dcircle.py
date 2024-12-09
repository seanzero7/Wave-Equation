import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

###############################################
#   WAVE EQUATION SIMULATION (2D)
#   Circular Domain
###############################################

#Parameters
L = 1.0            #Length of system
nx = 100           #Number of x points
ny = 100           #Number of y points
c = 1.0            #Wave speed
dx = L / (nx - 1)
dy = L / (ny - 1)
dt = 0.5 * min(dx, dy) / c  #Stable time step based on CFL

x = np.linspace(0, L, nx)
y = np.linspace(0, L, ny)
X, Y = np.meshgrid(x, y)

#Circular mask
radius = 0.5 * L
center_x, center_y = 0.5 * L, 0.5 * L
circular_mask = (X - center_x)**2 + (Y - center_y)**2 <= radius**2

#Gaussian pulse in the center within the circular mask
u0 = np.zeros_like(X)
u0[circular_mask] = np.exp(-((X[circular_mask] - center_x)**2 + (Y[circular_mask] - center_y)**2) / 0.01)

u_prev = np.copy(u0)
u_current = np.copy(u0)
u_prev[~circular_mask] = 0.0
u_current[~circular_mask] = 0.0

def wave_step_circular(u_current, u_prev, c, dx, dy, dt):
    u_next = np.zeros_like(u_current)
    #Computes second derivatives in x and y within the circular domain
    u_next[1:-1, 1:-1] = (2 * u_current[1:-1, 1:-1] - u_prev[1:-1, 1:-1] +
                          (c**2 * dt**2 / dx**2) * (u_current[2:, 1:-1] - 2 * u_current[1:-1, 1:-1] + u_current[:-2, 1:-1]) +
                          (c**2 * dt**2 / dy**2) * (u_current[1:-1, 2:] - 2 * u_current[1:-1, 1:-1] + u_current[1:-1, :-2]))
    u_next[~circular_mask] = 0.0  #Enforces the circular boundary condition
    return u_next

#Plots circular domain
fig_wave, ax_wave = plt.subplots()
im_wave = ax_wave.imshow(u_current, origin='lower', extent=[0, L, 0, L], vmin=-1, vmax=1, cmap='RdBu')
circle = plt.Circle((center_x, center_y), radius, color='black', fill=False, linewidth=2)
ax_wave.add_artist(circle)
ax_wave.set_title('2D Wave Equation Simulation on a Circular Domain')
ax_wave.set_xlabel('x')
ax_wave.set_ylabel('y')
fig_wave.colorbar(im_wave, ax=ax_wave, label='Displacement')

def update_wave_circular(frame):
    global u_current, u_prev
    u_next = wave_step_circular(u_current, u_prev, c, dx, dy, dt)
    u_prev = u_current
    u_current = u_next
    im_wave.set_data(u_current)
    return [im_wave]

#Animation
frames = 300 
fps = 20 
ani_wave = FuncAnimation(fig_wave, update_wave_circular, frames=frames, interval=1000 / fps, blit=True)

#Saving GIF
gif_filename = "wave_simulation_circular.gif"
writer = PillowWriter(fps=fps)
ani_wave.save(gif_filename, writer=writer)

print(f"GIF saved as {gif_filename}")
plt.show()
