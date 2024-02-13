from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as np
import jax
from functools import partial

import flux
import differentiation

@partial(jax.jit, static_argnums = (3,))
def split_form_spatial_standard_1D(u, gamma, D, numerical_flux_function):
    num_nodes_element = D.shape[0]
    u_r = np.reshape(u, (4, -1, num_nodes_element))
    u_r = np.transpose(u_r, (0,2,1))

    F = flux.flux_matrix_1D(u_r, gamma, numerical_flux_function)
    dF = differentiation.split_form_differentiate_standard_1D(D, F)

    return dF

_split_form_spatial_standard_x = jax.vmap(split_form_spatial_standard_1D, (1, None, None, None), (1))
_split_form_spatial_standard_y = jax.vmap(split_form_spatial_standard_1D, (2, None, None, None), (2))

@partial(jax.jit, static_argnums = (4))
def split_form_spatial_standard_x(u, gamma, weight_array, D, numerical_flux_function):
    #return _split_form_spatial_standard_x(u, gamma, num_nodes_element, D,  numerical_flux_function)
    return weight_array[None,:,:] * _split_form_spatial_standard_x(u, gamma, D, numerical_flux_function)

@partial(jax.jit, static_argnums = (4))
def split_form_spatial_standard_y(u, gamma, weight_array, D, numerical_flux_function):
    #return _split_form_spatial_standard_y(u, gamma, num_nodes_element, D,  numerical_flux_function)
    return weight_array[None,:,:] * _split_form_spatial_standard_y(u, gamma, D, numerical_flux_function)

@partial(jax.jit, static_argnums = (5))
def split_form_spatial_x(u, dy, gamma, weight_array, D, numerical_flux_function):
    return 0.5 * dy * split_form_spatial_standard_x(u, gamma, weight_array, D, numerical_flux_function)

@partial(jax.jit, static_argnums = (5))
def split_form_spatial_y(u, dx, gamma, weight_array, D, numerical_flux_function):
    return 0.5 * dx * split_form_spatial_standard_y(u, gamma, weight_array, D, numerical_flux_function)



if __name__ == "__main__":
    num_el = 5
    num_points = 4
    
    u = np.arange(4 * num_el * num_points)
    u = np.reshape(u, (4, num_el * num_points))
    print(u)

    u_r = np.reshape(u, (4, -1, num_points))
    u_r = np.transpose(u_r, (0,2,1))

    print(u_r)

    def num_flux(u1, u2, gamma):
        f1 = u1[0] + 2 * u2[0]
        f2 = 0.5 * u1[1] + u2[1]
        f3 = u1[2] + 0.25 * u2[2]
        f4 = 0.1 * u1[3] + 0.2 * u2[3]
        return np.array([f1,f2,f3,f4])
    
    F = flux.flux_matrix_1D(u_r, 1.4, num_flux)

    print(F)

    D = np.array([  [-1, 1, 0, 0], 
                    [-1, 0, 1, 0],
                    [0, -1, 0, 1],
                    [0, 0, -1, 1]   ])

    dF = split_form_spatial_standard_1D(u, 1.4, 4, D, num_flux)

    print(dF)

    u2 = u[:,None,:]
    u2 = np.concatenate((u2, u2), 1)

    dF2 = _split_form_spatial_standard_x(u2, 1.4, 4, D, num_flux)


    print(u2)
    print(dF2)

    