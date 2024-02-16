from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as np
import jax
from functools import partial

import functional

"""
Namespace for trace operations

TO DO:
- compile index arguments i_1 and i_2 as constants
"""

@partial(jax.jit, static_argnums = (4,5))
def trace_operator_standard_1D(u, gamma, i_1, i_2, numerical_flux_func, flux_func):
    """
    Computes trace operator values on a 1D domain of standard elements for values u with '-' and '+' trace values given at indices i_1 and i_2 resp. for
    a numerical and continuous flux numerical_flux_func and flux_func resp.

    It is assumed that u is padded with boundary conditions
    """
    i_1_pad = np.concatenate((i_1 + 1, np.array([-1])))
    i_2_pad = np.concatenate((np.array([0]), i_2 + 1))
    u1 = u[:,i_1_pad]
    u2 = u[:,i_2_pad]

    F = numerical_flux_func(u1,u2,gamma)
    f_1 = flux_func(u1, gamma)
    f_2 = flux_func(u2, gamma)

    dF1 = -(f_1 - F)
    dF2 = f_2 - F

    Fs = functional.interleave_vector(dF2,dF1) #np.reshape(np.vstack((dF1,dF2)), (-1,), order = 'F')
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

@partial(jax.jit, static_argnums = (7,8,9))
def trace_operator_x(u, dy, gamma, y, boundary_weights, j_1, j_2, numerical_flux_func, flux_func, boundary_func):
    """
    Computes trace operator values on '-' and '+' vertical boundaries at x-indices j_1 and j_2 respectively. Operations are carried out on a domain 
    of standard elements for a numerical and continuous flux given by numerical_flux_func and flux_func respectively. Function pads u with (y-dependent) 
    boundary conditions as given by boundary_func.

    boundary_weights is a 1D array with the weights of a trace quadrature rule for every standard element along the vertical boundary.
    """
    return 0.5 * dy * trace_operator_x_standard(u, gamma, y, boundary_weights, j_1, j_2, numerical_flux_func, flux_func, boundary_func)


@partial(jax.jit, static_argnums = (6,7,8))
def trace_operator_y_standard(u, gamma, x, boundary_weights, i_1, i_2, numerical_flux_func, flux_func, boundary_func):
    """
    Computes trace operator values on '-' and '+' horizontal boundaries at y-indices i_1 and i_2 respectively. Operations are carried out on a domain 
    of standard elements for a numerical and continuous flux given by numerical_flux_func and flux_func respectively. Function pads u with (x-dependent) 
    boundary conditions as given by boundary_func.

    boundary_weights is a 1D array with the weights of a trace quadrature rule for every standard element along the horizontal boundary.
    """

    u_pad = boundary_func(u, x)
    return boundary_weights[None,None,:] * _trace_operator_y_standard(u_pad, gamma, i_1, i_2, numerical_flux_func, flux_func)

@partial(jax.jit, static_argnums = (7,8,9))
def trace_operator_y(u, dx, gamma, x, boundary_weights, i_1, i_2, numerical_flux_func, flux_func, boundary_func):
    """
    Computes trace operator values on '-' and '+' horizontal boundaries at y-indices i_1 and i_2 respectively. Operations are carried out on a domain 
    of standard elements for a numerical and continuous flux given by numerical_flux_func and flux_func respectively. Function pads u with (x-dependent) 
    boundary conditions as given by boundary_func.

    boundary_weights is a 1D array with the weights of a trace quadrature rule for every standard element along the horizontal boundary.
    """
    return 0.5 * dx * trace_operator_y_standard(u, gamma, x, boundary_weights, i_1, i_2, numerical_flux_func, flux_func, boundary_func)



if __name__ == "__main__":

    '''
    arr = np.arange(3*3*3)
    arr = np.reshape(arr, (3,3,3))

    print(arr)

    vals = np.array([1,2,3])

    print(vals[None,None,:] * arr)

    '''
    import boundary

    num_el = 5
    num_points = 4

    i_1 = np.array([0, 4, 8, 12, 16], dtype = int)
    i_2 = np.array([3, 7, 11, 15, 19], dtype = int)
    
    i_1_pad = np.concatenate((i_1 + 1, np.array([-1])))
    i_2_pad = np.concatenate((np.array([0]), i_2 + 1))

    u = np.arange(4 * num_el * num_points)
    u = np.reshape(u, (4, num_el * num_points))
    u_pad = boundary.periodic_1D(u, 1)

    print(u_pad)

    def num_flux(u1, u2, gamma):
        f1 = 2 * u1[0] + 2 * u2[0]
        f2 = 0.5 * u1[1] + 0.5 * u2[1]
        f3 = 0.25 * u1[2] + 0.25 * u2[2]
        f4 = 0.1 * u1[3] + 0.1 * u2[3]
        return np.array([f1,f2,f3,f4])
    
    def cont_flux(u, gamma):
        return u
    
    dF = trace_operator_standard_1D(u_pad, 1.4, i_1, i_2, num_flux, cont_flux)
    
    print(dF)

    #print(i_1_pad)
    #print(i_2_pad)

    #print(u_pad[:,i_1_pad])
    #print(u_pad[:,i_2_pad])

    u2 = u_pad[:,None,:]
    u2 = np.concatenate((u2,u2), 1)

    print(u2)

    dF2 = _trace_operator_x_standard(u2, 1.4, i_1, i_2, num_flux, cont_flux)
    
    print(dF2)

    u3 = u[:,:,None]
    u3 = np.concatenate((u3,u3), 2)

    x = np.array([1,2])
    boundary_weights = np.array([1,2])

    dF3 = trace_operator_y_standard(u3, 1.4, x, boundary_weights, i_1, i_2, num_flux, cont_flux, boundary.periodic_y)
    
    print(dF3)
    