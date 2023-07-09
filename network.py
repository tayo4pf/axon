import numpy as np
import loss
import layer
import activation

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
        assert len(data.shape) == 2
        assert len(labels.shape) == 2
        losses = []

        #Iterate through the training data
        for subject, label in zip(data, labels):
            prop = subject.T
            label = label.T
            
            #Forward propagate the input
            for layer in self.layers:
                prop = layer.__forward(prop)

            #Calculate loss
            l = self.loss.loss(prop, label)
            losses.append(l)

            #Calculate differential of loss with respect to output values
            de_da = self.loss.partial_loss(prop, label)

            #Backpropogate differentials through network to update weights
            for layer in reversed(self.layers):
                de_da = layer.__backward(de_da, self.learning_rate)

        return np.array(losses)


    def test(self, data, labels):
        """
        Returns the of the model prediction losses for input data and labels
        @param data: 2-d numpy matrix of input subjects, each subject occupies a row
        @param labels: 2-d numpy matrix of labels, each label occupies row
        @output: numpy array of loss values for each subject
        """
        assert len(data.shape) == 2
        assert len(labels.shape) == 2
        losses = []

        prop = data.T
        for layer in self.layers:
            prop = layer.__forward(prop)

        prop = prop.T
        for output, target in zip(prop, labels):
            output = output.T
            target = target.T
            l = self.loss.loss(output, target)
            losses.append(l)
        
        return losses