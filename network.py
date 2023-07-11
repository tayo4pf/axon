import numpy as np
import loss
import layer
import activation
import matplotlib.pyplot as plt
import csv

class Network:
    def __init__(self, shape: list[int], activations: list[activation.Activation], loss: loss.Loss, learning_rate: float):
        assert len(shape) - 1 == len(activations)
        self.layers = [layer.Layer(shape[i], shape[i+1], activations[i]) for i in range(len(shape)-1)]
        self.shape = shape
        self.activations = activations
        self.learning_rate = learning_rate
        self.loss = loss

    def train(self, data: np.ndarray, labels: np.ndarray):
        """
        Trains the network using the given data and labels
        @param data: 2-d numpy matrix of input subjects, each subject occupies a row
        @param labels: 2-d numpy matrix of expected network output values, each label occupies a row
        labels rows should be the same width as the final layer output
        @output: numpy array of loss values for each subject
        """
        losses = []

        #Iterate through the training data
        for subject, label in zip(data, labels):
            prop = subject[np.newaxis].T
            label = label[np.newaxis].T
            #Forward propagate the input
            for layer in self.layers:
                prop = layer.forward(prop)

            #Calculate loss
            l = self.loss.loss(prop, label)
            losses.append(l)

            #Calculate differential of loss with respect to output values
            de_da = self.loss.partial_loss(prop, label)

            #Backpropogate differentials through network to update weights
            for layer in reversed(self.layers):
                de_da = layer.backward(de_da, self.learning_rate)

        return np.array(losses)


    def test(self, data, labels):
        """
        Returns the of the model prediction losses for input data and labels
        @param data: 2-d numpy matrix of input subjects, each subject occupies a row
        @param labels: 2-d numpy matrix of labels, each label occupies row
        @output: numpy array of loss values for each subject
        """
        losses = []

        for input, label in zip(data, labels):
            prop = input[np.newaxis].T
            label = label[np.newaxis].T
            for layer in self.layers:
                prop = layer.forward(prop)

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
                prop = layer.forward(prop)
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
        fieldnames = ['inputs', 'outputs', 'weights', 'biases', 'activation', 'learning_rate', 'loss']
        rows = []

        for layer in self.layers:
            row = {}
            row['inputs'] = layer.shape[1]
            row['outputs'] = layer.shape[0]
            row['weights'] = layer.weights.flatten()
            row['biases'] = layer.biases.flatten()
            row['activation'] = layer.activation.name()
            row['learning_rate'] = self.learning_rate
            row['loss'] = self.loss.name()
            rows.append(row)

        with open(filename+'.csv', 'w') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def read_from(self, filename):
        def str_activation(name):
            match name:
                case "IDENTITY":
                    return activation.Identity
                case "RELU":
                    return activation.Relu
                case "LEAKYRELU":
                    return activation.LeakyRelu
                case "SOFTMAX":
                    return activation.Softmax
            raise ModuleNotFoundError("Activation function specified in csv cannot be found:", name)
        
        def str_loss(name):
            match name:
                case "MSE":
                    return loss.MSE
                case "LOGISTIC":
                    return loss.Logistic
            raise ModuleNotFoundError("Loss function specified in csv cannot be found:", name)
        
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
                a = str_activation(row['activation'])
                loz = str_loss(row['loss'])
                learning_rate = float(row['learning_rate'])
                weights = np.array(row['weights'][1:-1].replace('\n','').split(), dtype = np.float64)
                biases = np.array(row['biases'][1:-1].replace('\n','').split(), dtype = np.float64)

                if i == 0:
                    self.shape.append(inputs)
                    i += 1
                self.shape.append(outputs)
                self.activations.append(a)
                self.loss = loz
                self.learning_rate = learning_rate

                l = layer.Layer(inputs, outputs, a)
                l.weights = np.reshape(weights, (outputs, inputs))
                l.biases = np.reshape(biases, (outputs, 1))
                l.shape = (outputs, inputs)

                self.layers.append(l)