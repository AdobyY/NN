import numpy as np

class DenseLayer:
    def __init__(self, n_units, activation):
        self.n_units = n_units
        self.activation = activation
        self.weights = None
        self.biases = None
    
    def initialize(self, input_size):
        self.weights = np.random.randn(input_size, self.n_units) * 0.01
        self.biases = np.zeros((1, self.n_units))


    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
        return self.activation.forward(self.z)

    def backward(self, dA):
        dZ = dA * self.activation.backward(self.z)
        m = self.inputs.shape[0]
        self.dW = np.dot(self.inputs.T, dZ) / m
        self.db = np.sum(dZ, axis=0, keepdims=True) / m
        return np.dot(dZ, self.weights.T)

    def update(self, optimizer):
        optimizer.update(self)

    def set_weights(self, weights):
        self.weights = weights

    def set_biases(self, bias):
        self.bias = bias
