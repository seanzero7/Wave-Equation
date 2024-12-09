import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#Domain parameters
nx, ny, nz = 20, 20, 20  #grid points
dx = 0.05                #spatial step
dy = dx                  #spatial step in y-direction (uniform grid)
dz = dx                  #spatial step in z-direction (uniform grid)
c = 1.0                  #wave speed (normalized)
dt = 0.9 * dx / (np.sqrt(3) * c)  #time step
steps = 200              #number of time steps to animate

#3D grid
x = np.linspace(0, (nx - 1) * dx, nx)
y = np.linspace(0, (ny - 1) * dx, ny)
z = np.linspace(0, (nz - 1) * dx, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

#Fields
E = np.zeros((nx, ny, nz, 3))  # Electric field
B = np.zeros((nx, ny, nz, 3))  # Magnetic field

#Initial condition: Gaussian pulse in Ex component
cx, cy, cz = nx // 2, ny // 2, nz // 2
sigma = 0.1
E[:, :, :, 0] = np.exp(-((X - X[cx, cy, cz])**2 + (Y - Y[cx, cy, cz])**2 + (Z - Z[cx, cy, cz])**2) / (2 * sigma**2))

#Curl function
def curl(field):
    Fx = field[:, :, :, 0]
    Fy = field[:, :, :, 1]
    Fz = field[:, :, :, 2]

    dFzdy = (Fz[:, 2:, :] - Fz[:, :-2, :]) / (2 * dy)
    dFydz = (Fy[:, :, 2:] - Fy[:, :, :-2]) / (2 * dz)
    dFxdz = (Fx[:, :, 2:] - Fx[:, :, :-2]) / (2 * dz)
    dFzdx = (Fz[2:, :, :] - Fz[:-2, :, :]) / (2 * dx)
    dFydx = (Fy[2:, :, :] - Fy[:-2, :, :]) / (2 * dx)
    dFxdy = (Fx[:, 2:, :] - Fx[:, :-2, :]) / (2 * dy)

    curl_x = np.zeros_like(Fx)
    curl_y = np.zeros_like(Fx)
    curl_z = np.zeros_like(Fx)

    curl_x[:, 1:-1, 1:-1] = dFzdy[:, :, 1:-1] - dFydz[:, 1:-1, :]
    curl_y[1:-1, :, 1:-1] = dFxdz[1:-1, :, :] - dFzdx[:, :, 1:-1]
    curl_z[1:-1, 1:-1, :] = dFydx[:, 1:-1, :] - dFxdy[1:-1, :, :]

    return np.stack((curl_x, curl_y, curl_z), axis=-1)

#Updates function for fields
def step(E, B):
    c2 = c**2
    curlE = curl(E)
    curlB = curl(B)

    E_new = E + dt * c2 * curlB
    B_new = B - dt * curlE

    #Dirichlet boundary conditions (fields vanish at boundaries)
    E_new[[0, -1], :, :, :] = 0
    E_new[:, [0, -1], :, :] = 0
    E_new[:, :, [0, -1], :] = 0

    B_new[[0, -1], :, :, :] = 0
    B_new[:, [0, -1], :, :] = 0
    B_new[:, :, [0, -1], :] = 0

    return E_new, B_new

#Visualization
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

slice_z = nz // 2

def init():
    ax.clear()
    Ex_slice = E[:, :, slice_z, 0]
    surf = ax.plot_surface(X[:, :, slice_z], Y[:, :, slice_z], Ex_slice, cmap='viridis')
    ax.set_zlim(-1, 1)
    return surf,

def update(frame):
    global E, B
    E, B = step(E, B)
    ax.clear()
    Ex_slice = E[:, :, slice_z, 0]
    surf = ax.plot_surface(X[:, :, slice_z], Y[:, :, slice_z], Ex_slice, cmap='viridis')
    ax.set_zlim(-1, 1)
    return surf,

ani = FuncAnimation(fig, update, frames=30, init_func=init, blit=False, interval=50)
ani.save('EMwave_equation.gif', writer='imagemagick', fps=8)

plt.show()
