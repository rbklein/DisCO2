from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as np
import jax
from functools import partial

import interior_ops
import trace_ops

@partial(jax.jit, static_argnums = (6,7,17,18,19,20,21,22,23,24))
def dudt(u, gamma, x, y, dx, dy, total_x, total_y, weight_array, weight_x, weight_y, D, 
         trace_inds, trace_x_i, trace_x_j, trace_y_i, trace_y_j, numerical_body_flux_x, 
         numerical_body_flux_y, numerical_trace_flux_x, numerical_trace_flux_y, 
         continuous_flux_x, continuous_flux_y, boundary_x, boundary_y):
    dudt = np.zeros((4, total_y, total_x))
    dudt -= interior_ops.split_form_spatial_x(u, dy ,gamma, weight_array, D, numerical_body_flux_x)
    dudt -= interior_ops.split_form_spatial_y(u, dx, gamma, weight_array, D, numerical_body_flux_y)

    trace_x = trace_ops.trace_operator_x(u, dy, gamma, y, weight_x, trace_inds["j_L"][0,:], trace_inds["j_R"][0,:], numerical_trace_flux_x, continuous_flux_x, boundary_x)
    trace_y = trace_ops.trace_operator_y(u, dx, gamma, x, weight_y, trace_inds["i_D"][:,0], trace_inds["i_U"][:,0], numerical_trace_flux_y, continuous_flux_y, boundary_y)

    dudt = dudt.at[:,trace_x_i,trace_x_j].add(trace_x)
    dudt = dudt.at[:,trace_y_i,trace_y_j].add(trace_y)

    dudt = dudt / (weight_array[None,:,:] * dx * dy * 0.25)
    return dudt































