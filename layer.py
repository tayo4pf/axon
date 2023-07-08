import numpy as np
import activation.py

class Layer:
    def __init__(self, inputs: int, outputs: int, activation: Activation):
        """
        Constructs layer object
        Generates weights and biases from input and output sizes
        @param inputs: number of input features for layer
        @
        """
        self.weights = np.random.rand(outputs, inputs)
        self.biases = np.random.rand(outputs, 1)

    def __forward(self, input):
        """
        forward input through the layer using weights and biases
        """
        self.input = input
        self.z = (np.matmul(self.weights, input), self.biases)
        self.a = self.activation.sigma(self.z)
        return self.a
    
    def __backward(self, input):
        