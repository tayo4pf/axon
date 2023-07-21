import numpy as np
import activation
from optimizer import Optimizer

class Layer:
    def __init__(self, inputs: int, outputs: int, activation=activation.Identity):
        """
        Constructs layer object
        Generates weights and biases from input and output sizes
        @param inputs: number of input features for layer
        @
        """
        self.weights = np.random.rand(outputs, inputs).astype(np.float64)
        self.weights = self.weights / np.sum(self.weights, axis = 1)[np.newaxis].T
        self.shape = (outputs, inputs)
        self.biases = np.random.rand(outputs, 1).astype(np.float64)
        self.activation = activation

        #Momentum attributes
        self.weights_velocity = np.zeros(self.weights.shape).astype(np.float64)
        self.biases_velocity = np.zeros(self.biases.shape).astype(np.float64)

        #Adaptive learning rate attribute
        self.ada_weights = np.zeros(self.weights.shape).astype(np.float64)
        self.ada_biases = np.zeros(self.biases.shape).astype(np.float64)

    def freeze(self):
        """
        Sets velocity (momentum) of derivatives to zero 
        Use to kill momentum between training sets
        """
        self.weights_velocity = np.zeros(self.weights.shape)
        self.biases_velocity = np.zeros(self.biases.shape)

    def reset_ada(self):
        """
        Sets adaptive learning rate components to zero
        Use to reset adaptive learning rate between training sets
        """
        self.ada_weights = np.zeros(self.weights.shape)
        self.ada_biases = np.zeros(self.biases.shape)

    def derivatives(self, partials):
        """
        Given derivative of error with respect to layer outputs,
        computes derivative with respect to weights and biases
        returns derivative with respect to inputs
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
        
        return de_dx

    def forward(self, optimizer, input, momentum):
        """
        Forwards input through network using method corresponding with optimizer
        @param optimizer: Optimizer to be used for forward
        @param input: Input data
        @param momentum: Momentum float between 0 and 1 to be used if optimizer requires momentum value
        """
        match optimizer:
            case Optimizer.NAG:
                return self.forward_nag(input, momentum)
        return self.forward_(input)

    def backward(self, optimizer: Optimizer, partials: np.ndarray, learning_rate: float, momentum: float, epsilon: float, gamma: float):
        """
        Performs backpropogation on layer with optimizer to modify weights and biases
        @param optimizer: Optimizer to be used for backpropagation
        @param partials: Derivative of error with respect to output values
        @param learning_rate: Learning rate value
        @param momentum: Momentum preservation used in momentum based optimizers - float between 0 and 1
        @param epsilon: Small number (default 0.01) used for adaptive learning rate
        @param gamma: Exponential averaging variable used - float between 0 and 1
        """
        match optimizer:
            case Optimizer.SGD:
                return self.backward_sgd(partials, learning_rate)
            case Optimizer.SGD_WM:
                return self.backward_sgd_wm(partials, learning_rate, momentum)
            case Optimizer.NAG:
                return self.backward_nag(partials, learning_rate, momentum)
            case Optimizer.AdaGrad:
                return self.backward_adagrad(partials, learning_rate, epsilon)
            case Optimizer.AdaDelta:
                return self.backward_adadelta(partials, learning_rate, gamma, epsilon)
            case Optimizer.AdaM:
                return self.backward_adam(partials, learning_rate, momentum, gamma, epsilon)

    def forward_(self, input):
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
    
    def backward_sgd(self, partials, learning_rate):
        """
        Performs backpropogation with stochastic gradient descent to modify the weights and biases using the input partials
        @param partials: is a vector of the derivative of the error with respect to the output values
        """
        de_dx = self.derivatives(partials)

        #Update weights and biases
        self.weights = self.weights - (learning_rate * self.de_dw)
        self.biases = self.biases - (learning_rate * self.de_db)
        return de_dx
    
    def backward_sgd_wm(self, partials, learning_rate, momentum):
        """
        Performs backpropagation with stochastic gradient descent with momentum
        @param partials: vector of the derivative of the error with respect to the output values
        @param momentum: float value between 0 and 1 for the proportion of derivative conserved 
        """
        de_dx = self.derivatives(partials)

        #Update velocity
        self.weights_velocity = (momentum * self.weights_velocity) + (learning_rate * self.de_dw)
        self.biases_velocity = (momentum * self.biases_velocity) + (learning_rate * self.de_db)

        #Update weights and biases
        self.weights = self.weights - self.weights_velocity
        self.biases = self.biases - self.biases_velocity
        return de_dx
    
    def forward_nag(self, input, momentum):
        """
        forward non-batch input through the layer using the NAG future features for training
        input is 2d array of shape (features, 1)
        """
        assert input.shape == (self.shape[1], 1)

        future_weights = self.weights - (momentum * self.weights_velocity)
        future_biases = self.biases - (momentum * self.biases_velocity)

        self.input = input
        self.z = np.add(np.matmul(future_weights, input), future_biases)
        assert self.z.shape == (self.shape[0], 1)

        self.a = self.activation.sigma(self.z)
        assert self.a.shape == self.z.shape

        return self.a
    
    def backward_nag(self, partials, learning_rate, momentum):
        """
        Performs backpropagation with nesterov accelerated gradient descent
        @param partials: vector of the derivative of the error with respect to the output values
        @param momentum: float value between 0 and 1 for the proportion of derivative conserved 
        """
        assert partials.shape == (self.shape[0], 1)

        #Compute accelerated weights
        future_weights = self.weights - (momentum * self.weights_velocity)

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
        
        #Generate de_dx partial vector for previous layer with accelerated weight values
        dz_dx = future_weights.T
        de_dx = np.matmul(dz_dx, de_dz)

        #Update velocity
        self.weights_velocity = (momentum * self.weights_velocity) + (learning_rate * self.de_dw)
        self.biases_velocity = (momentum * self.biases_velocity) + (learning_rate * self.de_db)

        #Update weights and biases
        self.weights = self.weights - self.weights_velocity
        self.biases = self.biases - self.biases_velocity
        return de_dx
    
    def backward_adagrad(self, partials, learning_rate, epsilon):
        """
        Performs backpropogation with adaptive gradient descent to modify the weights and biases using the input partials
        @param partials: is a vector of the derivative of the error with respect to the output values
        """
        de_dx = self.derivatives(partials)

        #Update weights and biases
        self.weights = self.weights - (learning_rate * self.de_dw/(np.sqrt(self.ada_weights + epsilon)))
        self.biases = self.biases - (learning_rate * self.de_db/(np.sqrt(self.ada_biases + epsilon)))
        self.ada_weights = self.ada_weights + np.square(self.de_dw)
        self.ada_biases = self.ada_biases + np.square(self.de_db)
        return de_dx
    
    def backward_adadelta(self, partials, learning_rate, gamma, epsilon):
        """
        Performs backpropagation with adaptive delta to modify the weights and biases using the input partials
        @param partials: is a vector of the derivative of the error with respect to the output values
        @param gamma: float between 0 and 1
        """
        de_dx = self.derivatives(partials)

        #Update weights and biases
        self.weights = self.weights - (learning_rate * self.de_dw/(np.sqrt(self.ada_weights + epsilon)))
        self.biases = self.biases - (learning_rate * self.de_db/(np.sqrt(self.ada_biases + epsilon)))
        self.ada_weights = (gamma * self.ada_weights) + ((1 - gamma) * np.square(self.de_dw))
        self.ada_biases = (gamma * self.ada_biases) + ((1 - gamma) * np.square(self.de_db))

        return de_dx
    
    def backward_adam(self, partials, learning_rate, momentum, gamma, epsilon):
        """
        Performs backpropagation with adam optimization to modify the weights and biases using the input partials
        @param partials: vector of the derivative of the error with respect to the output values
        @param momentum: momentum exponential decay variable
        @param gamma: adaptive learning rate exponential decay variable
        """
        de_dx = self.derivatives(partials)

        #Update weights and biases
        self.weights_velocity = (momentum * self.weights_velocity) + ((1 - momentum) * self.de_dw)
        self.biases_velocity = (momentum * self.biases_velocity) + ((1 - momentum) * self.de_db)
        self.weights = self.weights - (learning_rate * self.weights_velocity/(np.sqrt(self.ada_weights + epsilon)))
        self.biases = self.biases - (learning_rate * self.biases_velocity/(np.sqrt(self.ada_biases + epsilon)))
        self.ada_weights = (gamma * self.ada_weights) + ((1 - gamma) * np.square(self.de_dw))
        self.ada_biases = (gamma * self.ada_biases) + ((1 - gamma) * np.square(self.de_db))
        
        return de_dx