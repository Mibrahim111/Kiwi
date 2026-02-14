import numpy as np
from activation import ActivationFunction

class Sigmoid(ActivationFunction.ActivationFunction) : 
    def forward(self, activation):
        """
        Sigmoid activation Function
        """
        return 1.0 / (1.0 + np.exp(-activation))
    
    def derivative(self, output):
        return output * (1.0 - output)