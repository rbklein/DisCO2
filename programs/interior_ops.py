from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as np
import jax
from functools import partial

import flux
import differentiation

@partial(jax.jit, static_argnums = (2,4,))
def split_form_spatial_standard_1D(u, gamma, num_nodes_element, D,  numerical_flux_function):
    #print(u.shape)
    u_r = np.reshape(u, (4, -1, num_nodes_element))
    #print(u_r.shape)
    u_r = np.transpose(u_r, (0,2,1))

    F = flux.flux_matrix_1D(u_r, gamma, numerical_flux_function)
    dF = differentiation.split_form_differentiate_standard_1D(D, F)

    return dF

_split_form_spatial_standard_x = jax.vmap(split_form_spatial_standard_1D, (1, None, None, None, None), (1))
_split_form_spatial_standard_y = jax.vmap(split_form_spatial_standard_1D, (2, None, None, None, None), (2))

@partial(jax.jit, static_argnums = (3,5,))
def split_form_spatial_standard_x(u, gamma, weight_array, num_nodes_element, D,  numerical_flux_function):
    #print(u.shape)
    return weight_array[None,:,:] * _split_form_spatial_standard_x(u, gamma, num_nodes_element, D,  numerical_flux_function)

@partial(jax.jit, static_argnums = (3,5,))
def split_form_spatial_standard_y(u, gamma, weight_array, num_nodes_element, D,  numerical_flux_function):
    return weight_array[None,:,:] * _split_form_spatial_standard_y(u, gamma, num_nodes_element, D,  numerical_flux_function)

