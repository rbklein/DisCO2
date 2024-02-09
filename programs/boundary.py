from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as np
import jax

"""
Namespace for boundary condition functions
"""

@jax.jit
def periodic_1D(u, x):
    """
    Pads u periodically
    """
    return np.hstack((u[:,-1][:,None], u, u[:,0][:,None]))

periodic_y = jax.vmap(periodic_1D, (2,None), 2)
periodic_x = jax.vmap(periodic_1D, (1,None), 1)

if __name__ == "__main__":
    u = np.array([  
                    [     
                        [1,2,3],
                        [4,5,6],
                        [7,8,9]
                                    ],
                    [   
                        [1,2,3],
                        [4,5,6],
                        [7,8,9]
                                    ]   
                                        ])

    print(u)
    print(periodic_y(u, 1))