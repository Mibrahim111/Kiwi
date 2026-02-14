from abc import ABCMeta, abstractmethod

class ActivationFunction: # using Abstraction (strategy design pattern)
    '''
    classdocs
    '''
    __metaclass__ = ABCMeta
    
    @abstractmethod 
    def forward(self, activation):
        '''
        Transfer neuron activation is Second step of forward propagation.
        Once a neuron is activated, we need to transfer the activation to see what the neuron output actually is.
        '''
        raise NotImplementedError()
    
    @abstractmethod
    def derivative(self, output):
        '''
        Calculate the derivative of an neuron output.
        Given an output value from a neuron, we need to calculate it's local gradient.
        '''
        raise NotImplementedError()