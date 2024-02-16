from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as np
import matplotlib.pyplot as plt

import mesh
import quadrature
import differentiation
import flux
import boundary
import spatial
import temporal

degree = 2
num_points = degree + 1

Lx = 1
Ly = 1

Nelx = 256
Nely = 256

dx = Lx / Nelx
dy = Ly / Nely

dt = 0.001
T = 0.1
nT = T // dt

total_x = num_points * Nelx
total_y = num_points * Nely

X, Y, trace_inds, trace_x_i, trace_x_j, trace_y_i, trace_y_j = mesh.generate_mesh(degree, Lx, Ly, Nelx, Nely)

I = np.arange(total_y)[:,None] * np.ones(total_x)[None,:]
J = np.ones(total_y)[:,None] * np.arange(total_x)[None,:]

x = X[0,:]
y = Y[:,0]

gamma = 1.4


initial_rho = lambda x, y: 2 + 0.5 * np.exp(-100 * (y - 0.5)**2)  + 0 * x 
initial_u = lambda x, y: 0 * x + 0 * y   
initial_v = lambda x, y: 1/10 * np.exp(-100 * (y - 0.5)**2) + 0 * x        
initial_p = lambda x, y: initial_rho(x, y)**gamma
'''
initial_rho = lambda x, y: 2 + 0.5 * np.exp(-100 * (x - 0.5)**2)  + 0 * y 
initial_u = lambda x, y: 1/10 * np.exp(-100 * (x - 0.5)**2) + 0 * y 
initial_v = lambda x, y: 0 * x + 0 * y          
initial_p = lambda x, y: initial_rho(x, y)**gamma
'''

rho0 = initial_rho(X,Y)
u0 = initial_u(X,Y)
v0 = initial_v(X,Y)
p0 = initial_p(X,Y)

u = np.vstack((rho0[None,:,:], (rho0 * u0)[None,:,:], (rho0 * v0)[None,:,:], (p0 / (gamma - 1) + 0.5 * rho0 * u0**2)[None,:,:]))

tolerance = 0.0001
points, weights = quadrature.gauss_legendre_lobatto_quadrature(num_points, tolerance)
D = differentiation.differentiation_matrix_1D(points)
weight_array = quadrature.mass_array(num_points, tolerance, Nelx, Nely)
weight_x = quadrature.boundary_mass_array(num_points, tolerance, Nely)
weight_y = quadrature.boundary_mass_array(num_points, tolerance, Nelx)

it = 0
while it < nT:
    u = temporal.RK4(u, gamma, x, y, dx, dy, total_x, total_y, weight_array, weight_x, weight_y, D, 
         trace_inds, trace_x_i, trace_x_j, trace_y_i, trace_y_j, flux.ismail_roe_x, 
         flux.ismail_roe_y, flux.ismail_roe_x, flux.ismail_roe_y, flux.continuous_flux_x, flux.continuous_flux_y, boundary.periodic_x, boundary.periodic_y, spatial.dudt, dt)
    
    it += 1
    print('t: ', it * dt, ' it: ', it)
    
dudt_ = spatial.dudt(u, gamma, x, y, dx, dy, total_x, total_y, weight_array, weight_x, weight_y, D, 
         trace_inds, trace_x_i, trace_x_j, trace_y_i, trace_y_j, flux.ismail_roe_x, 
         flux.ismail_roe_y, flux.ismail_roe_x, flux.ismail_roe_y, flux.continuous_flux_x, flux.continuous_flux_y, boundary.periodic_x, boundary.periodic_y)

#import interior_ops
#dudt_ = interior_ops._split_form_spatial_standard_y(u, gamma, D, flux.ismail_roe_y)

fig, ax = plt.subplots()
ax.imshow(dudt_[0,:,:], origin = 'lower', interpolation='none')

fig, ax = plt.subplots(2,2)
ax[0,0].imshow(u[0,:,:], origin = 'lower', interpolation='none')
ax[0,0].set_title(r'$\rho$')

ax[0,1].imshow(u[1,:,:], origin = 'lower', interpolation='none')
ax[0,1].set_title(r'$\rho u$')

ax[1,0].imshow(u[2,:,:], origin = 'lower', interpolation='none')
ax[1,0].set_title(r'$\rho v$')

ax[1,1].imshow(u[3,:,:], origin = 'lower', interpolation='none')
ax[1,1].set_title(r'$E$')

plt.show()