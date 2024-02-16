from jax import config
config.update("jax_enable_x64", True)

import jax
from functools import partial


@partial(jax.jit, static_argnums = (6,7,17,18,19,20,21,22,23,24,25))
def RK4(u, gamma, x, y, dx, dy, total_x, total_y, weight_array, weight_x, weight_y, D, 
         trace_inds, trace_x_i, trace_x_j, trace_y_i, trace_y_j, numerical_body_flux_x, 
         numerical_body_flux_y, numerical_trace_flux_x, numerical_trace_flux_y, 
         continuous_flux_x, continuous_flux_y, boundary_x, boundary_y, dudt, dt):
    k1 = dudt(u, gamma, x, y, dx, dy, total_x, total_y, weight_array, weight_x, weight_y, D, 
         trace_inds, trace_x_i, trace_x_j, trace_y_i, trace_y_j, numerical_body_flux_x, 
         numerical_body_flux_y, numerical_trace_flux_x, numerical_trace_flux_y, 
         continuous_flux_x, continuous_flux_y, boundary_x, boundary_y)
    k2 = dudt(u + dt / 2 * k1, gamma, x, y, dx, dy, total_x, total_y, weight_array, weight_x, weight_y, D, 
         trace_inds, trace_x_i, trace_x_j, trace_y_i, trace_y_j, numerical_body_flux_x, 
         numerical_body_flux_y, numerical_trace_flux_x, numerical_trace_flux_y, 
         continuous_flux_x, continuous_flux_y, boundary_x, boundary_y)
    k3 = dudt(u + dt / 2 * k2, gamma, x, y, dx, dy, total_x, total_y, weight_array, weight_x, weight_y, D, 
         trace_inds, trace_x_i, trace_x_j, trace_y_i, trace_y_j, numerical_body_flux_x, 
         numerical_body_flux_y, numerical_trace_flux_x, numerical_trace_flux_y, 
         continuous_flux_x, continuous_flux_y, boundary_x, boundary_y)
    k4 = dudt(u + dt * k3, gamma, x, y, dx, dy, total_x, total_y, weight_array, weight_x, weight_y, D, 
         trace_inds, trace_x_i, trace_x_j, trace_y_i, trace_y_j, numerical_body_flux_x, 
         numerical_body_flux_y, numerical_trace_flux_x, numerical_trace_flux_y, 
         continuous_flux_x, continuous_flux_y, boundary_x, boundary_y)
    return u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)