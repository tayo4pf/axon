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
        return np.maximum(0, z)
    
    def partial_sigma(z: np.ndarray, a: np.ndarray):
        """
        Returns the hessian matrix of relu function output with respect to the input z
        """
        return np.diagflat(np.where(z > 0, 1, 0))


class Softmax(Activation):
    def sigma(z: np.ndarray):
        """
        Returns the input with the softmax function applied to it
        Only for 1 subject - not for batches
        """
        return np.exp(z) / np.sum(np.exp(z), axis = 0)


    def partial_sigma(z: np.ndarray, a: np.ndarray):
        """
        Returns the hessian matrix of softmax function output with respect to the input z
        Only for 1 subject, not batch
        """
        pre_hessian = -np.matmul(a, a.T)
        diagonal = np.diagflat(a)
        return np.add(diagonal, pre_hessian)
    
class Identity(Activation):
    def sigma(z: np.ndarray):
        """
        Returns the input with the identity function applied to it
        """
        return z
    
    def partial_sigma(z: np.ndarray, a: np.ndarray):
        """
        Returns the hessian matrix of identity function output with respect to the input z
        """
        return np.eye(len(z))