import numpy as np
from activation import ActivationFunction

class Softmax(ActivationFunction.ActivationFunction):

    def forward(self, activation):
        '''
        Softmax activation function.
        '''
        exp_values = np.exp(activation - np.max(activation, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities
    
    def derivative(self, output):
        # The derivative of softmax will be computed during backpropagation 
        pass