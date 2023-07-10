import numpy as np
import activation

class Layer:
    def __init__(self, inputs: int, outputs: int, activation=activation.Identity()):
        """
        Constructs layer object
        Generates weights and biases from input and output sizes
        @param inputs: number of input features for layer
        @
        """
        self.weights = np.random.rand(outputs, inputs)
        self.biases = np.random.rand(outputs, 1)
        self.activation = activation

    def forward(self, input):
        """
        forward non-batch input through the layer using weights and biases
        input is 2d array of shape (features, 1)
        """
        self.input = input
        self.z = np.add(np.matmul(self.weights, input), self.biases)
        assert self.z.shape[0] == len(self.weights)
        assert self.z.shape[1] == 1
        self.a = self.activation.sigma(self.z)
        return self.a
    
    def backward(self, partials, learning_rate):
        """
        performs backpropogation to modify the weights and biases using the input partials
        partials is a vector of the derivative of the error with respect to the output values
        """
        #Generate da_dz matrix
        da_dz = self.activation.partial_sigma(self.z, self.a)
        #Generate dz_dw matrix - several rows of the input
        dz_dw = np.tile(self.input.T, (len(self.weights), 1))
        #Generate de_dz vector
        assert da_dz.shape[0] == partials.shape[0]
        de_dz = np.matmul(da_dz, partials)
        self.de_dw = de_dz * dz_dw
        self.de_db = de_dz
        
        #Generate de_dx partial vector for previous layer
        de_dx = np.matmul(self.weights.T, de_dz)

        #Update weights and biases
        self.weights = self.weights - (learning_rate * self.de_dw)
        self.biases = self.biases - (learning_rate * self.de_db)
        return de_dx