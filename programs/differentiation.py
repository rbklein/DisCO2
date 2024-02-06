from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as np
import polynomials

"""
Namespace for differentiation operators
"""

def differentiation_matrix_1D(points):
    n = len(points)
    D = np.zeros((n,n))
    for i in range(n):
        D = D.at[:,i].set(polynomials.lagrange_polynomial_derivative(points, points, i))
    return D


if __name__ == "__main__":
    import quadrature

    p = quadrature.gauss_legendre_lobatto_points(7, 0.00001)
    D = differentiation_matrix_1D(p)

    f = lambda x: 3*x**6 + 2*x**2 + x + 5
    fp = lambda x: 18*x**5 + 4*x + 1

    y = f(p)
    yapp = D @ y
    yp = fp(p)

    print(yp - yapp)
    #print(D)