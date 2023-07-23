import numpy as np

class Loss:
    def from_name(name):
        match name:
                case "MSE":
                    return MSE
                case "LOGISTIC":
                    return Logistic
        raise ModuleNotFoundError("Loss function specified in csv cannot be found:", name)

class MSE(Loss):
    def loss(output, target):
        """
        Returns the loss value of the output and target using mean squared error
        """
        assert output.shape == target.shape
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