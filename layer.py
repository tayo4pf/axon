import numpy as np
import activation

class Layer:
    def __init__(self, inputs: int, outputs: int, activation=activation.Identity):
        """
        Constructs layer object
        Generates weights and biases from input and output sizes
        @param inputs: number of input features for layer
        @
        """
        self.weights = np.random.rand(outputs, inputs).astype(np.float64)
        self.shape = (outputs, inputs)
        self.biases = np.random.rand(outputs, 1).astype(np.float64)
        self.activation = activation

    def forward(self, input):
        """
        forward non-batch input through the layer using weights and biases
        input is 2d array of shape (features, 1)
        """
        assert input.shape == (self.shape[1], 1)
        self.input = input
        self.z = np.add(np.matmul(self.weights, input), self.biases)
        assert self.z.shape == (self.shape[0], 1)
        self.a = self.activation.sigma(self.z)
        assert self.a.shape == self.z.shape
        return self.a
    
    def backward(self, partials, learning_rate):
        """
        performs backpropogation to modify the weights and biases using the input partials
        partials is a vector of the derivative of the error with respect to the output values
        """
        assert partials.shape == (self.shape[0], 1)

        #Generate da_dz matrix
        da_dz = self.activation.partial_sigma(self.z, self.a)
        assert da_dz.shape == (self.shape[0], self.shape[0])

        #Generate dz_dw matrix - several rows of the input
        dz_dw = np.tile(self.input.T, (len(self.weights), 1))
        assert dz_dw.shape == self.weights.shape

        #Generate de_dz vector
        de_dz = np.matmul(da_dz, partials)
        assert de_dz.shape == (partials.shape)

        self.de_dw = de_dz * dz_dw
        self.de_db = de_dz
        
        #Generate de_dx partial vector for previous layer
        dz_dx = self.weights.T
        de_dx = np.matmul(dz_dx, de_dz)

        #Update weights and biases
        self.weights = self.weights - (learning_rate * self.de_dw)
        self.biases = self.biases - (learning_rate * self.de_db)
        return de_dx