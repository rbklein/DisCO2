from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as np
import jax

import matplotlib.pyplot as plt

import mesh
import quadrature
import differentiation
import flux
import interior_ops
import boundary
import trace_ops

degree = 3
num_points = degree + 1

Lx = 1
Ly = 1

Nelx = 256
Nely = 256

total_x = num_points * Nelx
total_y = num_points * Nely

X, Y, trace_inds = mesh.generate_mesh(degree, Lx, Ly, Nelx, Nely)

x = X[0,:]
y = Y[:,0]

i_R = trace_inds["i_R"]
j_R = trace_inds["j_R"]
i_L = trace_inds["i_L"]
j_L = trace_inds["j_L"]
i_U = trace_inds["i_U"]
j_U = trace_inds["j_U"]
i_D = trace_inds["i_D"]
j_D = trace_inds["j_D"]

trace_x_i = np.zeros((i_R.shape[0], i_L.shape[1] + i_R.shape[1]), dtype = int)
trace_x_j = np.zeros((j_R.shape[0], j_L.shape[1] + j_R.shape[1]), dtype = int)
trace_y_i = np.zeros((i_D.shape[0] + i_U.shape[0], i_U.shape[1]), dtype = int)
trace_y_j = np.zeros((j_D.shape[0] + j_U.shape[0], j_U.shape[1]), dtype = int)
trace_x_i = trace_x_i.at[:,::2].set(i_L) 
trace_x_i = trace_x_i.at[:,1::2].set(i_R)
trace_x_j = trace_x_j.at[:,::2].set(j_L) 
trace_x_j = trace_x_j.at[:,1::2].set(j_R)
trace_y_i = trace_y_i.at[::2,:].set(i_D)
trace_y_i = trace_y_i.at[1::2,:].set(i_U)
trace_y_j = trace_y_j.at[::2,:].set(j_D)
trace_y_j = trace_y_j.at[1::2,:].set(j_U)

gamma = 1.4
initial_rho = lambda x, y: 2 + 0.5 * np.exp(-100 * (x - 0.5)**2)  + 0 * y 
initial_u = lambda x, y: 1/10 * np.exp(-100 * (x - 0.5)**2) + 0 * y 
initial_v = lambda x, y: 0 * x + 0 * y          
initial_p = lambda x, y: initial_rho(x, y)**gamma

rho0 = initial_rho(X,Y)
u0 = initial_u(X,Y)
v0 = initial_v(X,Y)
p0 = initial_p(X,Y)

u = np.vstack((rho0[None,:,:], (rho0 * u0)[None,:,:], (rho0 * v0)[None,:,:], (p0 / (gamma - 1) + 0.5 * rho0 * u0**2)[None,:,:]))

tolerance = 0.00001
points, weights = quadrature.gauss_legendre_lobatto_quadrature(num_points, tolerance)
D = differentiation.differentiation_matrix_1D(points)
weight_array = quadrature.mass_array(num_points, tolerance, Nelx, Nely)
weight_x = quadrature.boundary_mass_array(num_points, tolerance, Nely)
weight_y = quadrature.boundary_mass_array(num_points, tolerance, Nelx)

dudt = np.zeros((4, total_y, total_x))
dudt -= interior_ops.split_form_spatial_standard_x(u, gamma, weight_array, num_points, D, flux.ismail_roe_x)
dudt -= interior_ops.split_form_spatial_standard_y(u, gamma, weight_array, num_points, D, flux.ismail_roe_y)

trace_x = trace_ops.trace_operator_x_standard(u, gamma, y, weight_x, j_L[0,:], j_R[0,:], flux.ismail_roe_x, flux.continuous_flux_x, boundary.periodic_x)
trace_y = trace_ops.trace_operator_y_standard(u, gamma, x, weight_y, i_D[:,0], i_U[:,0], flux.ismail_roe_y, flux.continuous_flux_y, boundary.periodic_y)

dudt = dudt.at[:,trace_x_i,trace_x_j].add(trace_x)
dudt = dudt.at[:,trace_y_i,trace_y_j].add(trace_y)

#fig, ax = plt.subplots(subplot_kw = {"projection" : "3d"})
#ax.plot_surface(X,Y,dudt[1,:,:])

fig, ax = plt.subplots()
#ax.contourf(X,Y,dudt[0,:,:])
ax.imshow(dudt[3,:,:])

plt.show()



























