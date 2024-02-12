from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as np
import jax
from functools import partial

import functional

"""
Namespace for trace operations
"""

@partial(jax.jit, static_argnums = (4,5))
def trace_operator_standard_1D(u, gamma, i_1, i_2, numerical_flux_func, flux_func):
    """
    Computes trace operator values on a 1D domain of standard elements for values u with '-' and '+' trace values given at indices i_1 and i_2 resp. for
    a numerical and continuous flux numerical_flux_func and flux_func resp.

    It is assumed that u is padded with boundary conditions
    """
    i_1_pad = np.concatenate((np.array([0]), i_1 + 1))
    i_2_pad = np.concatenate((i_2 + 1, np.array([-1])))
    u1 = u[:,i_1_pad]
    u2 = u[:,i_2_pad]

    F = numerical_flux_func(u1,u2,gamma)
    f_1 = flux_func(u1, gamma)
    f_2 = flux_func(u2, gamma)

    dF1 = f_1 - F
    dF2 = -(f_2 - F)

    Fs = functional.interleave_vector(dF1,dF2) #np.reshape(np.vstack((dF1,dF2)), (-1,), order = 'F')
    return Fs[:,1:-1]

_trace_operator_x_standard = jax.vmap(trace_operator_standard_1D, (1, None, None, None, None, None), 1)
_trace_operator_y_standard = jax.vmap(trace_operator_standard_1D, (2, None, None, None, None, None), 2)

@partial(jax.jit, static_argnums = (6,7,8))
def trace_operator_x_standard(u, gamma, y, boundary_weights, j_1, j_2, numerical_flux_func, flux_func, boundary_func):
    """
    Computes trace operator values on '-' and '+' vertical boundaries at x-indices j_1 and j_2 respectively. Operations are carried out on a domain 
    of standard elements for a numerical and continuous flux given by numerical_flux_func and flux_func respectively. Function pads u with (y-dependent) 
    boundary conditions as given by boundary_func.

    boundary_weights is a 1D array with the weights of a trace quadrature rule for every standard element along the vertical boundary.
    """

    u_pad = boundary_func(u, y)
    return boundary_weights[None,:,None] * _trace_operator_x_standard(u_pad, gamma, j_1, j_2, numerical_flux_func, flux_func)

@partial(jax.jit, static_argnums = (6,7,8))
def trace_operator_y_standard(u, gamma, x, boundary_weights, i_1, i_2, numerical_flux_func, flux_func, boundary_func):
    """
    Computes trace operator values on '-' and '+' horizontal boundaries at y-indices i_1 and i_2 respectively. Operations are carried out on a domain 
    of standard elements for a numerical and continuous flux given by numerical_flux_func and flux_func respectively. Function pads u with (x-dependent) 
    boundary conditions as given by boundary_func.

    boundary_weights is a 1D array with the weights of a trace quadrature rule for every standard element along the horizontal boundary.
    """

    u_pad = boundary_func(u, x)
    return np.transpose(boundary_weights[None,:,None]  * np.transpose(_trace_operator_y_standard(u_pad, gamma, i_1, i_2, numerical_flux_func, flux_func), (0,2,1)), (0,2,1))


if __name__ == "__main__":
    import mesh
    import flux
    import boundary
    import quadrature
    
    degree = 3
    num_points = degree + 1
    Nelx = 64
    Nely = 64
    Lx = 1
    Ly = 1

    print("generating mesh")
    X,Y,inds = mesh.generate_mesh(degree, Lx, Ly, Nelx, Nely) 
    print("done")

    j_1 = inds["j_L"]
    j_2 = inds["j_R"]

    i_1 = inds["i_D"]
    i_2 = inds["i_U"]

    print(j_1.shape)
    print(j_2.shape)

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

    print(u.shape)

    print("generating quadrature")
    points, weights = quadrature.gauss_legendre_lobatto_quadrature(num_points, 0.00001)
    print("done")

    boundary_weights = np.tile(weights, Nely)

    print("calculating trace")
    trace_operator_x_standard(u, gamma, Y[:,0], boundary_weights, j_1[0,:], j_2[0,:], flux.ismail_roe_x, flux.continuous_flux_x, boundary.periodic_x)
    print("done")
    
    

