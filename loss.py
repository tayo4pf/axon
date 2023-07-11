import numpy as np

class Loss:
    def __init__(self):
        pass

    def loss(output, target):
        pass

    def partial_loss(output, target):
        pass

class MSE(Loss):
    def loss(output, target):
        """
        Returns the loss value of the output and target using mean squared error
        """
        assert output.shape == target.shape, (output, target)
        return np.sum(np.square(output - target))

    def partial_loss(output, target):
        """
        Returns the partial differential of the loss with respect to the output
        """
        assert output.shape == target.shape
        return -2 * (target - output)
    
    def name() -> str:
        return "MSE"
    
class Logistic(Loss):
    def loss(output, target):
        """
        Returns the loss value of the output and target using logistic error
        """
        assert output.shape == target.shape
        return -np.sum(target * np.log(output))
    
    def partial_loss(output, target):
        """
        Returns the partial differential of the loss with respect to the output
        """
        assert output.shape == target.shape
        return -(target/output)
    
    def name() -> str:
        return "LOGISTIC"