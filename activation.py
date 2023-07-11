from typing import Any
import numpy as np

class Activation:
    def __init__(self):
        pass

class Relu(Activation):
    def sigma(z: np.ndarray):
        """
        Returns the input with the relu function applied to it
        """
        return np.maximum(0, z)
    
    def partial_sigma(z: np.ndarray, a: np.ndarray):
        """
        Returns the hessian matrix of relu function output with respect to the input z
        """
        return np.diagflat(np.where(z > 0, 1, 0))
    
    def name() -> str:
        return "RELU"

class LeakyRelu(Activation):
    def sigma(z: np.ndarray):
        """
        Returns the input with the leaky relu function applied to it with coefficient 0.`
        """
        return np.maximum(0.1 * z, z)
    
    def partial_sigma(z: np.ndarray, a:np.ndarray):
        """
        Returns the hessian matrix of leaky relu function output with respect to the input z
        """
        return np.diagflat(np.where(z > 0, 1, 0.1 ))
    
    def name() -> str:
        return "LEAKYRELU"

class Softmax(Activation):
    def sigma(z: np.ndarray):
        """
        Returns the input with the softmax function applied to it
        """
        return np.exp(z) / np.sum(np.exp(z), axis = 0)


    def partial_sigma(z: np.ndarray, a: np.ndarray):
        """
        Returns the hessian matrix of softmax function output with respect to the input z
        """
        pre_hessian = -np.matmul(a, a.T)
        diagonal = np.diagflat(a)
        return np.add(diagonal, pre_hessian)
    
    def name() -> str:
        return "SOFTMAX"

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
    
    def name() -> str:
        return "IDENTITY"