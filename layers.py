import numpy as np

class DenseLayer:
    def __init__(self, size, activation):
        self.size = size
        self.activation = activation  # This will be a function now
        self.weights = None
        self.biases = None

    def build(self, input_size):
        self.weights = np.random.randn(input_size, self.size) * 0.01
        self.biases = np.zeros((1, self.size))

    def forward(self, inputs):
        self.inputs = inputs
        return self.activation(np.dot(inputs, self.weights) + self.biases)

    def get_params(self):
        return np.concatenate([self.weights.flatten(), self.biases.flatten()])
