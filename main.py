import numpy as np
from model import NeuralNetwork
from layers import DenseLayer
from activations import ReLU  # Import the function directly
from optimizers import SGDOptimizer

# Sample data
X = np.random.rand(100, 10)  # 100 samples, 10 features
y = np.random.rand(100, 1)    # 100 target values

# Create and train model
model = NeuralNetwork(X.shape[1])
model.add_layer(DenseLayer(8, ReLU))  # Pass the function directly, no parentheses
model.add_layer(DenseLayer(1, ReLU))  # Pass the function directly, no parentheses

optimizer = SGDOptimizer(learning_rate=0.01)
model.fit(X, y, epochs=1000, optimizer=optimizer)
