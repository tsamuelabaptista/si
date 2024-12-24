from abc import abstractmethod

import numpy as np


class Optimizer:

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        raise NotImplementedError

class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) in neural networks is an optimization algorithm that updates the model's parameters 
    based on the gradients of the loss function.

    SGD can be improved by adding a momentum term. This term accelerates the convergence by accumulating past gradients, 
    helping the model navigate through areas of high curvature in the loss landscape more efficiently.
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        """
        Initialize the SGD optimizer.

        Parameters
        ----------
        learning_rate: float
            The learning rate to use for updating the weights.
        momentum:
            The momentum to use for updating the weights.
        """
        # parameters
        super().__init__(learning_rate)
        self.momentum = momentum

        # attributes
        self.retained_gradient = None

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        # verify if self.retained_gradient is initialized
        # if not initialize it as a matrix of zeros
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros(np.shape(w))
        # compute and update the retained gradient
        self.retained_gradient = self.momentum * self.retained_gradient + (1 - self.momentum) * grad_loss_w
        # compute and update the new weights
        return w - self.learning_rate * self.retained_gradient
    
class Adam(Optimizer):
    """
    Adam can be looked at as a combination of RMSprop and SGD with momentum. 
    It uses the squared gradients to scale the learning rate like RMSprop 
    and it takes advantage of momentum by using moving average of the gradient 
    instead of gradient itself like SGD with momentum.
    """

    def __init__(self, learning_rate: float = 0.01, beta_1: float = 0.9, beta_2: float = 0.999, epsilon: float = 1e-8):
        """
        Initialize the Adam optimizer.

        Parameters
        ----------
        learning_rate: float
            The learning rate to use for updating the weights.
        beta_1: float
            The exponential decay rate for the 1st moment estimates.
        beta_2: float
            The exponential decay rate for the 2nd moment estimates.
        epsilon: float
            A small constant for numerical stability.
        """
        # parameters
        super().__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        # attributes
        self.m = None
        self.v = None
        self.t = 0

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        # verify if m and v are initialized
        # if not initialize them as matrices of zeros
        if self.m is None:
            self.m = np.zeros(np.shape(w))
        if self.v is None:
            self.v = np.zeros(np.shape(w))
        # update t (t+=1)
        self.t += 1 
        # compute and update m
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad_loss_w
        # compute and update v
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (grad_loss_w ** 2)
        # compute m_hat
        m_hat = self.m / (1 - self.beta_1)
        # compute v_hat
        v_hat = self.v / (1 - self.beta_2)
        # return the updated weights based on the moving averages
        return w - self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.epsilon))