from typing import Any
import numpy as np

class Activation:
    def __init__(self):
        pass

class Relu(Activation):
    def sigma(z: np.ndarray):
        """
        Returns the input with the relu function applied to it
        Only for 1 subject - not for batches
        """
        assert z.shape[1] == 1
        f = lambda x: 0 if x < 0 else x
        a = np.fromiter((f(xi) for xi in z), z.dtype, count = len(z))
        a = np.reshape(a, z.shape)
        return a
    
    def partial_sigma(z: np.ndarray, a: np.ndarray):
        """
        Returns the differential of relu function with respect to the input z
        """
        assert z.shape[1] == 1
        f = lambda x: 0 if x <= 0 else 1
        return np.diagflat(np.fromiter((f(xi) for xi in z), z.dtype, count = len(z)))


class Softmax(Activation):
    def sigma(z: np.ndarray):
        """
        Returns the input with the softmax function applied to it
        Only for 1 subject - not for batches
        """
        return np.exp(z) / np.sum(np.exp(z), axis = 0)


    def partial_sigma(z: np.ndarray, a: np.ndarray):
        """
        Returns the differential of relu function with respect to the input z
        Only for 1 subject, not batch
        """
        pre_hessian = -np.matmul(a, a.T)
        diagonal = np.diagflat(a)
        return np.add(diagonal, pre_hessian)