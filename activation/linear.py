import numpy as np
from activation import ActivationFunction

class Linear(ActivationFunction.ActivationFunction):

    def forward(self, activation):
        '''
        Linear activation function.
        '''
        return activation 
    
    def derivative(self, output):        
        return np.ones_like(output)