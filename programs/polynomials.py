import jax
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.linspace(-1, 1, 500)
    y = jacobi_polynomial_second_derivative(x, 2, 0, 0)

    plt.plot(x,y)
    plt.show()
    






