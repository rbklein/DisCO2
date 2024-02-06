from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as np

"""
Namespace for functions that calculate values of Jacobi polynomials and their derivatives
"""

def jacobi_polynomial(x, degree, a, b):
    """
    Evaluate the Jacobi polynomial with coefficients a,b of degree p = degree at x

    The value of the Jacobi polynomial is computed using the recursion relation at p 586 of Karniadakis
    """
    if degree < 0:
        raise ValueError('error: degree < 0 invalid order')
    if degree == 0:
        #returns an array in case x is an array
        return 1 + 0 * x
    if degree == 1:
        return 0.5 * (a - b + (a + b + 2) * x)
    
    n = degree - 1
    an1 = 2 * (n + 1) * (n + a + b + 1) * (2 * n + a + b)
    an2 = (2 * n + a + b + 1) * (a**2 - b**2)
    an3 = (2 * n + a + b) * (2 * n + a + b + 1) * (2 * n + a + b + 2)
    an4 = 2 * (n + a) * (n + b) * (2 * n + a + b + 2)

    return ((an2 + an3 * x) * jacobi_polynomial(x, n, a, b) - an4 * jacobi_polynomial(x, n - 1, a, b)) / an1

def jacobi_polynomial_derivative(x, degree, a, b):
    """
    Evaluate derivative of Jacobi polynomial with coefficients a,b of degree p = degree at x

    The value of the derivative is computed using relation (14) given at https://mathworld.wolfram.com/JacobiPolynomial.html
    """
    if degree < 0:
        raise ValueError('error: degree < 0 invalid order')
    if degree == 0:
        #returns an array in case x is an array
        return 0 * x
    
    return 0.5 * (degree + a + b + 1) * jacobi_polynomial(x, degree - 1, a + 1, b + 1)

def jacobi_polynomial_second_derivative(x, degree, a, b):
    """
    Evaluate second derivative of Jacobi polynomial with coefficients a,b of degree p = degree at x

    The value of the second derivative is computed by recursively applying relation (14) given at https://mathworld.wolfram.com/JacobiPolynomial.html
    """
    if degree < 0:
        raise ValueError('error: degree < 0 invalid order')
    if degree == 0:
        #returns an array in case x is an array
        return 0 * x
    if degree == 1:
        return 0 * x
    
    return 0.25 * (degree + a + b + 1) * (degree + a + b + 2) * jacobi_polynomial(x, degree - 2, a + 2, b + 2)
    
def legendre_polynomial(x, degree):
    """
    Evaluate the Legendre polynomial of degree p = degree at x

    Wrapper around jacobi_polynomial with a=b=0
    """
    return jacobi_polynomial(x, degree, 0, 0)

def legendre_polynomial_derivative(x, degree):
    """
    Evaluate derivative of Legendre polynomial of degree p = degree at x

    Wrapper around jacobi_polynomial_derivative with a=b=0
    """
    return jacobi_polynomial_derivative(x, degree, 0, 0)

def legendre_polynomial_second_derivative(x, degree):
    """
    Evaluate second derivative of Legendre polynomial of degree p = degree at x

    Wrapper around jacobi_polynomial_second_derivative with a=b=0
    """
    return jacobi_polynomial_second_derivative(x, degree, 0, 0)

def lagrange_polynomial(x, points, i):
    """
    Evaluate the unique Lagrange polynomial with value one in points[i] and zeros in points[j] with j not i at the point x

    Value is computed through straightforward computation
    """
    if i < 0:
        raise ValueError('error: positive index i required')
    if len(points) < 1:
        raise ValueError('error: at least one point is necessary')
    if len(points) <= i:
        raise ValueError('error: index i out of bounds for points array')
    if len(np.unique(points)) != len(points):
        raise ValueError('error: no duplicate values allowed')
    if len(points) == 1:
        return 1
    
    n = len(points)

    num = 1
    den = 1

    j = 0
    while j < n:
        if j != i:
            num *= (x - points[j])
            den *= (points[i] - points[j])
        j += 1

    return num / den

def lagrange_polynomial_derivative(x, points, i):
    """
    Evaluate the derivative of the unique Lagrange polynomial with value one in points[i] and zeros in points[j] with j not i at the point x

    Value is computed through straightforward computation
    """
    if i < 0:
        raise ValueError('error: positive index i required')
    if len(points) < 1:
        raise ValueError('error: at least one point is necessary')
    if len(points) <= i:
        raise ValueError('error: index i out of bounds for points array')
    if len(np.unique(points)) != len(points):
        raise ValueError('error: no duplicate values allowed')
    if len(points) == 1:
        return 0
    
    n = len(points)

    num = 0
    den = 1

    j = 0
    k = 0
    while j < n:
        if j != i:
            den *= (points[i] - points[j])
        j += 1

    while k < n:
        if i != k:
            new = 1
            j = 0
            while j < n:
                if j != i and j != k:
                    new *= (x - points[j])
                j += 1
            
            num += new
        k += 1
    
    return num / den



if __name__ == "__main__":

    import matplotlib.pyplot as plt

    #x = np.linspace(-1, 1, 500)
    #y = jacobi_polynomial_second_derivative(x, 2, 0, 0)

    import quadrature

    x = np.linspace(-1, 1, 500)
    #p = np.linspace(-1,1,5)
    p = quadrature.gauss_legendre_lobatto_points(4, 0.0001)
    yl = lagrange_polynomial(x, p, 2)
    yd = lagrange_polynomial_derivative(x, p, 2)

    plt.plot(x,yl)
    plt.plot(x,yd)
    plt.plot(p, np.zeros(len(p)), 'o')
    plt.show()
    






