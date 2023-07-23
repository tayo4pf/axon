import numpy as np
from axon import loss
from axon import layer
from axon import activation as activation
from axon.optimizer import Optimizer
import matplotlib.pyplot as plt
import csv

class Network:
    def __init__(self, shape: list[int], activations: list[activation.Activation], loss: loss.Loss):
        """
        Constructs a fresh neural network object
        @param shape: list[int] of number of inputs and outputs at each layer
        @param activations: list[activation.Activation] of activation modules for each layer of neurons
        @param loss: loss.Loss, loss module for the loss function of the network
        """
        assert len(shape) - 1 == len(activations)
        self.layers = [layer.Layer(shape[i], shape[i+1], activations[i]) for i in range(len(shape)-1)]
        self.shape = shape
        self.activations = activations
        self.loss = loss

    def train(self, data: np.ndarray, labels: np.ndarray, optimizer: Optimizer, learning_rate: float, momentum = 0.9, epsilon = 0.01, gamma = 0.8, fresh = True):
        """
        Trains the network using the given data and labels
        @param data: 2-d numpy matrix of input subjects, each subject occupies a row
        @param labels: 2-d numpy matrix of expected network output values, each label occupies a row
        labels rows should be the same width as the final layer output
        @param optimizer: Optimizer to be used for training
        @param learning_rate: Learning rate - float value
        @param momentum: Momentum value to be used for momentum based optimizers - float value between 0 and 1, default 0.9
        @param epsilon: Epsilon value for adaptive learning rate optimizers, small float value, default 0.01
        @param gamma: Exponential averaging value to be used for adaptive learning rate optimizers, float value between 0 and 1
        @param fresh: Boolean variable, when set to true all momentum and adaptive learning variables will be reset to zero
        @output: numpy array of loss values for each subject
        """
        losses = []

        assert momentum < 1
        assert momentum > 0
        assert gamma < 1
        assert gamma > 0

        if fresh:
            for layer in self.layers:
                layer.freeze()
                layer.reset_ada()

        if not isinstance(optimizer, Optimizer):
            raise TypeError("Optimizer must be an instance of Optimizer Enum")

        #Iterate through the training data
        for subject, label in zip(data, labels):
            prop = subject[np.newaxis].T
            label = label[np.newaxis].T
            #Forward propagate the input
            for layer in self.layers:
                prop = layer.forward(optimizer, prop, momentum)

            #Calculate loss
            l = self.loss.loss(prop, label)
            losses.append(l)

            #Calculate differential of loss with respect to output values
            de_da = self.loss.partial_loss(prop, label)

            #Backpropogate differentials through network to update weights
            for layer in reversed(self.layers):
                de_da = layer.backward(optimizer, de_da, learning_rate, momentum, epsilon, gamma)

        return np.array(losses)


    def test(self, data, labels):
        """
        Returns the of the model prediction losses for input data and labels
        @param data: 2-d numpy matrix of input subjects, each subject occupies a row
        @param labels: 2-d numpy matrix of labels, each label occupies row
        @output: numpy array of loss values for each subject
        """
        losses = []

        #Iterate through dataset
        for input, label in zip(data, labels):
            #Fix dimensions for input and label
            prop = input[np.newaxis].T
            label = label[np.newaxis].T

            #Forward propagate the input
            for layer in self.layers:
                prop = layer.forward(Optimizer.SGD, prop, None)

            #Calculate loss
            l = self.loss.loss(prop, label)
            losses.append(l)
        
        return np.array(losses)
    
    def predict(self, data):
        """
        Returns the model prediction of given input data
        @param data: 2-d numpy matrix of input subjects, each subject occupies a row
        """
        preds = []
        for input in data:
            prop = input[np.newaxis].T
            for layer in self.layers:
                prop = layer.forward(Optimizer.SGD, prop, None)
            preds.append(prop)
        
        return np.array(preds)
    
    def visualize(self):
        """
        Displays a heatmap representation of each layers weights and biases in the model
        """
        fig, graphs = plt.subplots(nrows = len(self.layers), 
                                   ncols = 2, 
                                   width_ratios = [max(self.shape[:-1]), 1],
                                   height_ratios = self.shape[1:])
        for i, (layer, graph_row) in enumerate(zip(self.layers, graphs)):
            weight_graph = graph_row[0].imshow(layer.weights, cmap='hot')
            graph_row[0].set_title("Weights")
            graph_row[0].set_xticklabels([])
            graph_row[0].set_yticklabels([])
            graph_row[0].tick_params(left=False, bottom=False)
            bias_graph = graph_row[1].imshow(layer.biases, cmap='hot')
            graph_row[1].set_title("Biases")
            graph_row[1].set_xticklabels([])
            graph_row[1].set_yticklabels([])
            graph_row[1].tick_params(left=False, bottom=False)
        fig.tight_layout()
        plt.show()

    def write_to(self, filename):
        fieldnames = ['inputs', 'outputs', 'weights', 'biases', 'weights_velocity', 'biases_velocity', 'ada_weights', 'ada_biases', 'activation', 'loss']
        rows = []

        for layer in self.layers:
            row = {}
            row['inputs'] = layer.shape[1]
            row['outputs'] = layer.shape[0]
            row['weights'] = layer.weights.flatten()
            row['biases'] = layer.biases.flatten()
            row['weights_velocity'] = layer.weights_velocity.flatten()
            row['biases_velocity'] = layer.biases_velocity.flatten()
            row['ada_weights'] = layer.ada_weights.flatten()
            row['ada_biases'] = layer.ada_biases.flatten()
            row['activation'] = layer.activation.name()
            row['loss'] = self.loss.name()
            rows.append(row)

        with open(filename+'.csv', 'w') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def __read_from(self, filename):        
        self.shape = []
        self.layers = []
        self.activations = []
        self.learning_rate = None
        self.loss = None
        with open(filename+'.csv') as f:
            reader = csv.DictReader(f)
            i = 0
            for row in reader:

                inputs = int(row['inputs'])
                outputs = int(row['outputs'])
                a = activation.Activation.from_name(row['activation'])
                loz = loss.Loss.from_name(row['loss'])
                weights = np.array(row['weights'][1:-1].replace('\n','').split(), dtype = np.float64)
                biases = np.array(row['biases'][1:-1].replace('\n','').split(), dtype = np.float64)
                weights_velocity = np.array(row['weights_velocity'][1:-1].replace('\n','').split(), dtype = np.float64)
                biases_velocity = np.array(row['biases_velocity'][1:-1].replace('\n','').split(), dtype = np.float64)
                ada_weights = np.array(row['ada_weights'][1:-1].replace('\n','').split(), dtype = np.float64)
                ada_biases = np.array(row['ada_biases'][1:-1].replace('\n','').split(), dtype = np.float64)

                if i == 0:
                    self.shape.append(inputs)
                    i += 1
                self.shape.append(outputs)
                self.activations.append(a)
                self.loss = loz

                l = layer.Layer(inputs, outputs, a)
                l.weights = np.reshape(weights, (outputs, inputs))
                l.biases = np.reshape(biases, (outputs, 1))
                l.weights_velocity = np.reshape(weights_velocity, (outputs, inputs))
                l.biases_velocity = np.reshape(biases_velocity, (outputs, 1))
                l.ada_weights = np.reshape(ada_weights, (outputs, inputs))
                l.ada_biases = np.reshape(ada_biases, (outputs, 1))
                l.shape = (outputs, inputs)

                self.layers.append(l)

    def read(filename):
        nn = Network([None], [], None)
        nn.__read_from(filename)
        return nn