import numpy as np

class DenseLayer:
    def __init__(self, num_inputs, num_neurons, activation_func):
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.activation_func = activation_func
        self.weights = np.random.uniform(-1, 1, (num_neurons, num_inputs))
        self.biases = np.random.uniform(-1, 1, num_neurons)
        self.output = np.zeros(num_neurons)

    def forward(self, inputs):
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)  # Перетворюємо на (1, num_inputs)
        z = np.dot(inputs, self.weights.T) + self.biases  # Розмірність (batch_size, num_neurons)
        self.output = np.array([self.activation_func.activate(x) for x in z.flatten()]).reshape(z.shape)
        return self.output

    def mutate(self, mutation_rate):
        mutation_mask_w = np.random.rand(*self.weights.shape) < mutation_rate
        mutation_values_w = np.random.uniform(-0.5, 0.5, self.weights.shape)
        self.weights += mutation_mask_w * mutation_values_w

        mutation_mask_b = np.random.rand(*self.biases.shape) < mutation_rate
        mutation_values_b = np.random.uniform(-0.5, 0.5, self.biases.shape)
        self.biases += mutation_mask_b * mutation_values_b

    def get_weights(self):
        return self.weights, self.biases