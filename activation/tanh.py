import numpy as np
from activation import ActivationFunction

class Tanh(ActivationFunction.ActivationFunction):

    def forward(self, activation):
        '''
        Tanh activation function.
        '''
        return np.tanh(activation)
    
    def derivative(self, output):
        return 1.0 - (output * output)