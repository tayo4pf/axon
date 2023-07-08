from typing import Any
import numpy as np

class Activation:
    def __init__(self):
        pass

class Relu(Activation):
    def __init__(self):
        pass

    def sigma(z: np.ndarray):
        """
        Returns the input with the relu function applied to it
        Only for 1 subject - not for batches
        """
        f = lambda x: 0 if x < 0 else x
        return np.fromiter((f(xi) for xi in z), z.dtype, count = len(z))
    
    def partial_sigma(z: np.ndarray, a: np.ndarray):
        """
        Returns the differential of relu function with respect to the  
        """
        pre_hessian = -np.matmul(a, a.T)
        diagonal = np.diag(a)
        return np.add(diagonal, pre_hessian)