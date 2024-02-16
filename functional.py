from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as np
import jax

"""
Namespace for general functions of practical use
"""

def interleave(arr1, arr2):
    """
    Interleave values of two 1D arrays arr1 and arr2

    i.e. [1,3,5] and [2,4,6] give [1,2,3,4,5,6]
    """
    return np.reshape(np.vstack((arr1,arr2)), (-1,), order = 'F')

interleave_vector = jax.vmap(interleave, (0,0), 0)