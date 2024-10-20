import numpy as np
from losses import MSE

class NeuralNetwork:
    def __init__(self, input_size):
        self.layers = []
        self.input_size = input_size

    def add_layer(self, layer):
        if len(self.layers) == 0:
            layer.build(self.input_size)
        else:
            layer.build(self.layers[-1].size)
        self.layers.append(layer)

    def predict(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def fit(self, X, y, epochs=1000, optimizer=None, loss_fn=MSE):
        weights = np.concatenate([layer.get_params() for layer in self.layers])
        
        for epoch in range(epochs):
            predictions = self.predict(X)
            loss = loss_fn(y, predictions)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss}')

            # Update weights using optimizer
            weights = optimizer.apply(loss_fn, X, y, weights)
