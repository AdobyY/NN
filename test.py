import numpy as np

class ReLU:
    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x):
        return (x > 0).astype(float)

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

import numpy as np

class MSE:
    @staticmethod
    def forward(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def backward(y_true, y_pred):
        return -2 * (y_true - y_pred) / y_true.size

class NN:
    def __init__(self, input_size):
        self.layers = []
        self.input_size = input_size  # Зберігаємо вхідний розмір

    def add_layer(self, layer):
        if self.layers:
            layer.initialize(self.layers[-1].n_units)  # Передаємо розмір попереднього шару
        else:
            layer.initialize(self.input_size)  # Вхідний розмір
        self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, y_true, y_pred, loss_fn):
        dA = loss_fn.backward(y_true, y_pred)
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def fit(self, X, y, epochs, optimizer, loss_fn, verbose=False):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = loss_fn.forward(y, y_pred)
            self.backward(y, y_pred, loss_fn)
            for layer in self.layers:
                layer.update(optimizer)
            if epoch % 100 == 0 and verbose:
                print(f'Epoch {epoch}, Loss: {loss}')

class GDOptimizer:
    def __init__(self, eta=0.01):
        self.eta = eta

    def update(self, layer):
        layer.weights -= self.eta * layer.dW
        layer.biases -= self.eta * layer.db

