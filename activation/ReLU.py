import numpy as np
from activation import ActivationFunction

class ReLU(ActivationFunction.ActivationFunction): 
    def forward(self, activation):
        """ 
        Rectified Linear activation function 
        """ 
        return np.maximum(0, activation)
        
    def derivative(self, output):
        return np.where(output <= 0, 0, 1)
