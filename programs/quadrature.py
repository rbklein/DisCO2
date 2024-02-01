import jax.numpy as np
import polynomials

"""
Namespace for functions calculating high-order quadrature points and weights

See:

https://phys.libretexts.org/Bookshelves/Astronomy__Cosmology/Celestial_Mechanics_(Tatum)/01%3A_Numerical_Methods/1.16%3A_Gaussian_Quadrature_-_Derivation#:~:text=It%20follows%20that%20the%20Gaussian,function%20f(x)%20obtained%20by 
    
for simple derivation of general Gaussian quadrature points

"""


def chebyshev_points(num_points):
    """
    Computes n = num_points Chebyshev points using analytical formula

    Formula is given at p 598 of Karniadakis
    """
    inds = np.arange(0,num_points)
    return -np.cos((2 * inds + 1) / (2 * num_points) * np.pi)

def gauss_legendre_lobatto_points(num_points, tolerance):
    """
    Computes n = num_points Gauss-Legendre-Lobatto (GLL) points iteratively to specified stopping criterion 'tolerance'

    Points are computed using a simple Newton iteration process with polynomial deflation and initial guess for the k-th points is taken 
    as k-th chebyshev point. Similar algorithm is described at p 598 of Karniadakis with P_m^(a,b) replaced with 
    d/dx(P_(num_points-1)^(0,0)) = d/dx(L_(num_points-1)) with L_n a Legendre polynomial of degree n. Necessary derivatives are present in polynomials.py.
    """
    if num_points < 2:
        raise ValueError('error: minimum of 2 gauss-legendre-lobatto points required')
    
    if num_points == 2:
        return np.array([-1,1])

    points = chebyshev_points(num_points)
    points = points.at[0].set(-1).at[-1].set(1)

    for i in range(1,num_points-1):
        r = points[i] #0.5 * (points[i] + points[i-1])
        eps = tolerance
        while True:
            s = 0
            for j in range(i-1):
                s += 1/(r - points[j]) 
            s += 1/(r - points[-1])
            delta = -polynomials.legendre_polynomial_derivative(r, num_points-1) / (polynomials.legendre_polynomial_second_derivative(r, num_points-1) - s * polynomials.legendre_polynomial_derivative(r, num_points-1))
            r += delta
            if np.abs(delta) < eps:
                break
        points = points.at[i].set(r)

    return points

def gauss_legendre_lobatto_quadrature(num_points, tolerance):
    """
    Computes n = num_points Gauss-Legendre-Lobatto (GLL) points and weights, iterations are carried out to specified stopping criterion 'tolerance'

    Points are computed using the function gauss_legendre_lobatto_points and weights are computed using analytical formula at:
    https://mathworld.wolfram.com/LobattoQuadrature.html
    """

    points = gauss_legendre_lobatto_points(num_points, tolerance)
    weights = 2 / (num_points * (num_points - 1)) * np.ones(num_points)
    vals = []
    for i in range(1,num_points-1):
        vals.append(2 / (num_points * (num_points - 1) * polynomials.legendre_polynomial(points[i], num_points - 1)**2))
    
    weights = weights.at[1:num_points-1].set(vals)
    return points, weights
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    n = 9

    pnts_cheb = chebyshev_points(n)
    pnts_gll = gauss_legendre_lobatto_points(n, 0.00001)

    p, weights = gauss_legendre_lobatto_quadrature(n, 0.00001)

    print(pnts_cheb)
    print(pnts_gll)
    print(p)
    print(weights)

    plt.plot(pnts_cheb, np.zeros(n), 'o')
    plt.plot(pnts_gll, np.zeros(n), 'o')

    plt.show()
