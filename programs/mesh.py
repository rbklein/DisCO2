from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as np 
import quadrature

"""
Namespace for meshing related functions

TO DO:
- boundary padded indices
"""

def xofsigma(sigma, a, b):
    """
    Transforms standard interval to [a,b]
    """
    return 0.5 * (b - a) * sigma + 0.5 * (b + a)

def sigmaofx(x, a, b):
    """
    Transforms [a,b] into standard interval
    """
    return 2 / (b - a) * x - (b + a) / (b - a)

def generate_mesh(degree, Lx, Ly, Nelx, Nely):
    """
    Generates two arrays containing mesh points for domain [0,Lx] x [0,Ly] with Nelx elements along x axis and Nely elements along y axis
    the elements contain n = degree + 1 Gauss Legendre Lobatto points. A dictionary containing grid indices of boundary points is also generated
    """
    dx = Lx / Nelx
    dy = Ly / Nely

    num_points = degree + 1
    points, _ = quadrature.gauss_legendre_lobatto_quadrature(num_points, 0.00001)

    num_total_x = Nelx * num_points
    num_total_y = Nely * num_points

    meshx = np.zeros(num_total_x)
    meshy = np.zeros(num_total_y)

    boundary_inds_i_L = np.arange(0,num_total_y)
    boundary_inds_j_L = np.zeros((Nelx))

    boundary_inds_i_R = np.arange(0,num_total_y)
    boundary_inds_j_R = np.zeros((Nelx))

    boundary_inds_i_U = np.zeros((Nely)) 
    boundary_inds_j_U = np.arange(0,num_total_x)

    boundary_inds_i_D = np.zeros((Nely))
    boundary_inds_j_D = np.arange(0,num_total_x)

    for i in range(Nelx):
        meshx = meshx.at[i * num_points:(i + 1) * num_points].set(xofsigma(points, i * dx, (i + 1) * dx))
        boundary_inds_j_L = boundary_inds_j_L.at[i].set(i * num_points)
        boundary_inds_j_R = boundary_inds_j_R.at[i].set((i + 1) * num_points - 1)

    for i in range(Nely):
        meshy = meshy.at[i * num_points:(i + 1) * num_points].set(xofsigma(points, i * dy, (i + 1) * dy))
        boundary_inds_i_U = boundary_inds_i_U.at[i].set(i * num_points)
        boundary_inds_i_D = boundary_inds_i_D.at[i].set((i + 1) * num_points - 1)


    X = np.ones(num_total_y)[:,None] * meshx[None,:]
    Y =  meshy[:,None] * np.ones(num_total_x)[None,:]

    boundary_inds_j_L = (np.ones(num_total_y)[:,None] * boundary_inds_j_L[None,:]).astype(int)
    boundary_inds_j_R = (np.ones(num_total_y)[:,None] * boundary_inds_j_R[None,:]).astype(int)
    boundary_inds_i_L = (boundary_inds_i_L[:,None] * np.ones(Nelx)[None,:]).astype(int)
    boundary_inds_i_R = (boundary_inds_i_R[:,None] * np.ones(Nelx)[None,:]).astype(int)

    boundary_inds_i_U = (boundary_inds_i_U[:,None] * np.ones(num_total_x)[None,:]).astype(int)
    boundary_inds_i_D = (boundary_inds_i_D[:,None] * np.ones(num_total_x)[None,:]).astype(int)
    boundary_inds_j_U = (np.ones(Nely)[:,None] * boundary_inds_j_U[None,:]).astype(int)
    boundary_inds_j_D = (np.ones(Nely)[:,None] * boundary_inds_j_D[None,:]).astype(int)

    boundary_inds = {'i_U' : boundary_inds_i_U, 'j_U' : boundary_inds_j_U, 'i_D' : boundary_inds_i_D, 'j_D' : boundary_inds_j_D, 'i_L' : boundary_inds_i_L, 'j_L' : boundary_inds_j_L, 'i_R' : boundary_inds_i_R, 'j_R' : boundary_inds_j_R}

    return X, Y, boundary_inds

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    X,Y,b_inds = generate_mesh(3, 1, 1, 3, 3)

    n = len(X[0,:])

    fig, ax = plt.subplots()
    for i in range(n):
        for j in range(n):
            plt.plot(X[i,j], Y[i,j], 'o')

    print(X[b_inds['i_L'],b_inds['j_L']], Y[b_inds['i_L'],b_inds['j_L']])

    ticks = [0,1/3,2/3,1]
    ax.set_xticks(ticks, minor = False)
    ax.set_yticks(ticks, minor = False)
    ax.grid()

    plt.show()
