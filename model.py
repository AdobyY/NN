import pickle

class NeuralNetwork:
    def __init__(self, input_size):
        self.input_size = input_size
        self.layers = []

    def add_layer(self, layer):
        if not self.layers:
            if layer.num_inputs != self.input_size:
                raise ValueError("Перший шар має відповідати розміру вхідних даних.")
        else:
            if layer.num_inputs != self.layers[-1].num_neurons:
                raise ValueError("Кількість входів нового шару повинна відповідати кількості нейронів попереднього шару.")
        self.layers.append(layer)

    def forward(self, X):
        # X: (batch_size, input_size) або (input_size,)
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def fit(self, X, y, epochs, optimizer):
        best_network, best_fitness = optimizer.optimize(X, y, self, epochs)
        self.layers = best_network.layers
        return {'best_fitness': best_fitness}

    def predict(self, X):
        return self.forward(X)
    
    def save_weights(self, filepath):
        with open(filepath, 'wb') as f:
            layer_params = [(layer.weights, layer.biases) for layer in self.layers]
            pickle.dump(layer_params, f)

    def load_weights(self, filepath):
        with open(filepath, 'rb') as f:
            layer_params = pickle.load(f)
            for layer, (weights, biases) in zip(self.layers, layer_params):
                layer.weights = weights
                layer.biases = biases