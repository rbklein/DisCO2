from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as np
import jax

"""
Namespace for (numerical) flux functions
"""

@jax.jit
def continuous_flux_x(u, gamma):
    """
    Compute the continuous convective flux in the x-direction of the compressible Euler/Navier-Stokes equations
    """
    rho_e = u[3] - 0.5 * (u[1]**2 + u[2]**2) / u[0]  
    p = (gamma - 1) * rho_e

    f1 = u[1]
    f2 = u[1]**2 / u[0] + p
    f3 = u[1] * u[2] / u[0]
    f4 = (u[3] + p) * u[1] / u[0]

    return np.array([f1,f2,f3,f4])

@jax.jit
def continuous_flux_y(u, gamma):
    """
    Compute the continuous convective flux in the y-direction of the compressible Euler/Navier-Stokes equations
    """
    rho_e = u[3] - 0.5 * (u[1]**2 + u[2]**2) / u[0]  
    p = (gamma - 1) * rho_e

    f1 = u[2]
    f2 = u[1] * u[2] / u[0]   
    f3 = u[2]**2 / u[0] + p
    f4 = (u[3] + p) * u[1] / u[0]

    return np.array([f1,f2,f3,f4])

@jax.jit
def logmean(a, b):
    """
    Computes the logarithmic mean using Ismail and Roe's algorithm
    """
    d = a / b
    f = (d - 1) / (d + 1)
    u = f**2
    F = np.where(u < 0.001, 1 + u / 3 + u**2 / 5 + u**3 / 7, np.log(d) / 2 / f)
    return (a + b) / (2 * F)

@jax.jit
def ismail_roe_x(u1, u2, gamma):
    """
    Ismail and Roe's entropy-conserving numerical flux in the x-direction
    """
    rho_e1 = u1[3] - 0.5 * (u1[1]**2 + u1[2]**2) / u1[0]
    p1 = (gamma - 1) * rho_e1

    rho_e2 = u2[3] - 0.5 * (u2[1]**2 + u2[2]**2) / u2[0]
    p2 = (gamma - 1) * rho_e2

    z1 = np.ones(u1.shape)
    z1 = z1.at[1].set(u1[1] / u1[0])
    z1 = z1.at[2].set(u1[2] / u1[0])
    z1 = z1.at[3].set(p1)
    z1 = np.sqrt(u1[0]/p1) * z1

    z2 = np.ones(u2.shape)
    z2 = z2.at[1].set(u2[1] / u2[0])
    z2 = z2.at[2].set(u2[2] / u2[0])
    z2 = z2.at[3].set(p2)
    z2 = np.sqrt(u2[0]/p2) * z2

    z1m = 0.5 * (z1[0] + z2[0])
    z2m = 0.5 * (z1[1] + z2[1])
    z3m = 0.5 * (z1[2] + z2[2])
    z4m = 0.5 * (z1[3] + z2[3])
    z1ln = logmean(z1[0],z2[0])
    z4ln = logmean(z1[3],z2[3])

    F1 = z2m * z4ln
    F2 = z4m / z1m + z2m / z1m * F1
    F3 = z3m / z1m * F1
    #F4 = 0.5 * (z2m / z1m) * ((gamma + 1) / (gamma - 1) * (z3ln / z1ln) + F2)
    F4 = 1 / (2 * z1m) * ((gamma + 1) / (gamma - 1) * F1 / z1ln + z2m * F2 + z3m * F3)
    return np.array([F1,F2,F3,F4])

@jax.jit
def ismail_roe_y(u1, u2, gamma):
    """
    Ismail and Roe's entropy-conserving numerical flux in the y-direction
    """
    rho_e1 = u1[3] - 0.5 * (u1[1]**2 + u1[2]**2) / u1[0]
    p1 = (gamma - 1) * rho_e1

    rho_e2 = u2[3] - 0.5 * (u2[1]**2 + u2[2]**2) / u2[0]
    p2 = (gamma - 1) * rho_e2

    z1 = np.ones(u1.shape)
    z1 = z1.at[1].set(u1[1] / u1[0])
    z1 = z1.at[2].set(u1[2] / u1[0])
    z1 = z1.at[3].set(p1)
    z1 = np.sqrt(u1[0]/p1) * z1

    z2 = np.ones(u2.shape)
    z2 = z2.at[1].set(u2[1] / u2[0])
    z2 = z2.at[2].set(u2[2] / u2[0])
    z2 = z2.at[3].set(p2)
    z2 = np.sqrt(u2[0]/p2) * z2

    z1m = 0.5 * (z1[0] + z2[0])
    z2m = 0.5 * (z1[1] + z2[1])
    z3m = 0.5 * (z1[2] + z2[2])
    z4m = 0.5 * (z1[3] + z2[3])
    z1ln = logmean(z1[0],z2[0])
    z4ln = logmean(z1[3],z2[3])

    G1 = z3m * z4ln
    G2 = z2m / z1m * G1 
    G3 = z4m / z1m + z3m / z1m * G1
    #F4 = 0.5 * (z2m / z1m) * ((gamma + 1) / (gamma - 1) * (z3ln / z1ln) + F2)
    G4 = 1 / (2 * z1m) * ((gamma + 1) / (gamma - 1) * G1 / z1ln + z2m * G2 + z3m * G3)
    return np.array([G1,G2,G3,G4])



if __name__ == "__main__":
    arr_l = np.ones((3,1024,1024))
    arr_r = 2 * np.ones((3,1024,1024))

    print(ismail_roe_x(arr_l, arr_r, 1.4).shape)


