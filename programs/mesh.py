import jax.numpy as np 
import quadrature

def xofsigma(sigma, a, b):
    return 0.5 * (b-a) * sigma + 0.5 * (b + a)

def sigmaofx(x, a, b):
    return 2 / (b - a) * x - (b + a) / (b - a)

def generate_mesh(degree, Lx, Ly, Nelx, Nely):
    dx = Lx / Nelx
    dy = Ly / Nely

    num_points = degree + 1
    points, _ = quadrature.gauss_legendre_lobatto_quadrature(num_points, 0.00001)

    print(points)

    meshx = np.zeros(Nelx * num_points)
    meshy = np.zeros(Nely * num_points)

    for i in range(Nelx):
        meshx = meshx.at[i * num_points:(i + 1) * num_points].set(xofsigma(points, i * dx, (i + 1) * dx))
    for i in range(Nely):
        meshy = meshy.at[i * num_points:(i + 1) * num_points].set(xofsigma(points, i * dy, (i + 1) * dy))

    X = np.ones(Nely * num_points)[:,None] * meshx[None,:]
    Y =  meshy[:,None] * np.ones(Nelx * num_points)[None,:]

    return X, Y

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    X,Y = generate_mesh(3, 1, 1, 256, 256)

    #n = len(X[0,:])

    #print(X)

    #fig, ax = plt.subplots()
    #for i in range(n):
    #    for j in range(n):
    #        plt.plot(X[i,j], Y[i,j], 'o')

    #ticks = [0,1/3,2/3,1]
    #ax.set_xticks(ticks, minor = False)
    #ax.set_yticks(ticks, minor = False)
    #ax.grid()

    plt.show()
