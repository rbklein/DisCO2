from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as np
import jax
from functools import partial

import polynomials

"""
Namespace for differentiation operators
"""

def differentiation_matrix_1D(points):
    """
    Compute differentiation matrix of 1D Lagrange polynomials associated points i.e. D_ij = dh_j/dx(x_i) with h_j(x) the Lagrange polynomial 
    with value one in points[i] and zeros in points[j] with j not i
    """
    if len(points) < 1:
        raise ValueError('error: at least one point required')
    if len(points) == 1:
        return 0

    n = len(points)
    D = np.zeros((n,n))
    for i in range(n):
        D = D.at[:,i].set(polynomials.lagrange_polynomial_derivative(points, points, i))
    return D


@partial(jax.jit, static_argnums=(2,3))
def differentiate_standard_1D(u, D, num_values, num_nodes_element):
    """
    Differentiate 1D grid function u of length num_values on standard elements with num_nodes_element nodes using standard element differentiation matrix D

    Grid function u is reshaped to 2D array A with element values as columns and multiplied by D as D @ A to obtain 2D derivative array A' which is then 
    reshaped back to derivative grid function u' 

    Function is JIT-compiled and therefore does not check argument compatibility
    """
    u_vec = np.reshape(u, (num_nodes_element, -1), order = 'F')
    du_vec = D @ u_vec
    return np.reshape(du_vec, (num_values), order = 'F')

"""
How to add docstring to vmapped functions?
"""
differentiate_x_standard = jax.vmap(differentiate_standard_1D, (0,None,None,None), (0))
differentiate_y_standard = jax.vmap(differentiate_standard_1D, (1,None,None,None), (1))
    

def split_form_differentiate_standard_1D(numerical_flux_function):
    #generate array of flux_matrices with vmapped flux_matrix function taking u in R^4 as argument
    #(D * A) * 1
    pass

"""
TO DO:

h & p convergence tests
"""
if __name__ == "__main__":
    import mesh
    import quadrature
    import matplotlib.pyplot as plt

    degree = 3
    points = quadrature.gauss_legendre_lobatto_points(degree + 1, 0.00001)

    Nelx = 256
    Nely = 256
    Lx = 1
    Ly = 1
    X, Y, ind = mesh.generate_mesh(degree, Lx, Ly, Nelx, Nely)

    dx = Lx / Nelx
    dy = Ly / Nely

    f = lambda x, y: np.cos(4 * 2 * np.pi * x) * x**3 + 10 * y**3 * x
    fp = lambda x, y: 2 * np.cos(4 * 2 * np.pi * x) * x**2 - 4 * 2 * np.pi * np.sin(4 * 2 * np.pi * x) * x**3 + 10 * y**2
    u = f(X,Y)

    num_nodes_element = degree + 1
    num_values = Nelx * num_nodes_element
    D = differentiation_matrix_1D(points)

    dudx = 2 / dx * differentiate_x_standard(u, D, num_values, num_nodes_element)
    dudy = 2 / dx * differentiate_y_standard(u, D, num_values, num_nodes_element)

    #plt.figure()
    #plt.imshow(u)

    #plt.figure()
    #plt.imshow(du)

    #plt.figure()
    #plt.imshow(X,Y,fp(X,Y))
    
    fig, ax = plt.subplots(subplot_kw = {"projection" : "3d"})
    ax.plot_surface(X,Y,dudx)
    
    fig, ax = plt.subplots(subplot_kw = {"projection" : "3d"})
    ax.plot_surface(X,Y,dudy)

    plt.show()



    '''
    degree = 12
    points = quadrature.gauss_legendre_lobatto_points(degree + 1, 0.00001)

    Nelx = 256
    Nely = 5
    Lx = 1
    Ly = 1
    X, Y, ind = mesh.generate_mesh(degree, Lx, Ly, Nelx, Nely)
    x = X[0,:]

    dx = Lx / Nelx

    f = lambda x: np.cos(4 * 2 * np.pi * x) * x**3
    fp = lambda x: 2 * np.cos(4 * 2 * np.pi * x) * x**2 - 4 * 2 * np.pi * np.sin(4 * 2 * np.pi * x) * x**3
    u = f(x)

    num_nodes_element = degree + 1
    num_values = Nelx * num_nodes_element
    D = differentiation_matrix_1D(points)

    du = 2 / dx * differentiate_standard_1D(u, D, num_values, num_nodes_element)

    plt.figure()
    plt.plot(x, np.zeros(len(x)), '-+')
    plt.plot(x, u)

    plt.figure()
    plt.plot(x, du)
    plt.plot(x, fp(x))
    plt.show()
    '''


    '''
    import quadrature

    p = quadrature.gauss_legendre_lobatto_points(7, 0.00001)
    D = differentiation_matrix_1D(p)

    f = lambda x: 3*x**6 + 2*x**2 + x + 5
    fp = lambda x: 18*x**5 + 4*x + 1

    y = f(p)
    yapp = D @ y
    yp = fp(p)

    print(yp - yapp)
    print(D)
    '''