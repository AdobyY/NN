import numpy as np
import pickle
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class ReLU:
    def forward(self, x):
        return np.maximum(0, x)
    
    def backward(self, x):
        return np.where(x > 0, 1, 0)

class Sigmoid:
    def forward(self, x):
        return 1 / (1 + np.exp(-x))
    
    def backward(self, x):
        s = self.forward(x)
        return s * (1 - s)

class Linear:
    def forward(self, x):
        return x
    
    def backward(self, x):
        return 1

class DenseLayer:
    def __init__(self, num_inputs, num_neurons, activation_func=None):
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.activation_func = activation_func if activation_func else Linear()
        self.weights = np.random.uniform(-1, 1, (num_neurons, num_inputs))
        self.biases = np.random.uniform(-1, 1, num_neurons)
        
        # Storage for forward pass
        self.inputs = None
        self.z = None  # Pre-activation
        self.output = None  # Post-activation
        
        # Storage for gradients
        self.dweights = None
        self.dbiases = None
        
    def forward(self, inputs):
        self.inputs = inputs
        if inputs.ndim == 1:
            self.inputs = inputs.reshape(1, -1)
            
        self.z = np.dot(self.inputs, self.weights.T) + self.biases
        self.output = self.activation_func.forward(self.z)
        return self.output
    
    def backward(self, gradient):
        # Calculate gradients through activation function
        d_activation = self.activation_func.backward(self.z) * gradient
        
        # Calculate gradients w.r.t weights and biases
        self.dweights = np.dot(d_activation.T, self.inputs)
        self.dbiases = np.sum(d_activation, axis=0)
        
        # Calculate gradient for next layer
        return np.dot(d_activation, self.weights)

class GradientDescentOptimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.epsilon = 1e-8  # Для числової стабільності
        
    def normalize_gradients(self, gradients):
        # Нормалізація градієнтів за допомогою L2 норми
        grad_norm = np.sqrt(np.sum(gradients * gradients) + self.epsilon)
        return gradients / grad_norm if grad_norm > self.epsilon else gradients
        
    def update_params(self, layer):
        # Нормалізуємо градієнти
        normalized_dweights = self.normalize_gradients(layer.dweights)
        normalized_dbiases = self.normalize_gradients(layer.dbiases)
        
        # Застосовуємо градієнтний спуск з нормалізованими градієнтами
        layer.weights -= self.learning_rate * normalized_dweights
        layer.biases -= self.learning_rate * normalized_dbiases

class NeuralNetwork:
    def __init__(self, input_size):
        self.input_size = input_size
        self.layers = []
        
    def add_layer(self, layer):
        if not self.layers:
            if layer.num_inputs != self.input_size:
                raise ValueError("First layer must match input size")
        else:
            if layer.num_inputs != self.layers[-1].num_neurons:
                raise ValueError("Layer input size must match previous layer output size")
        self.layers.append(layer)
        
    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, y_true, y_pred):
        batch_size = y_true.shape[0]
        # MSE derivative
        gradient = 2 * (y_pred - y_true) / batch_size
        
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
    
    def train_step(self, X, y, optimizer):
        # Forward pass
        predictions = self.forward(X)
        
        # Backward pass
        self.backward(y, predictions)
        
        # Update weights
        for layer in self.layers:
            optimizer.update_params(layer)
        
        # Calculate loss
        loss = np.mean((predictions - y) ** 2)
        return loss
    
    def fit(self, X, y, epochs=1000, learning_rate=0.01, verbose=True):
        optimizer = GradientDescentOptimizer(learning_rate=learning_rate)
        history = []
        
        for epoch in range(epochs):
            loss = self.train_step(X, y, optimizer)
            history.append(loss)
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
                
        return history

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


df = sm.datasets.macrodata.load_pandas().data

# Функція для створення дати з року та кварталу
def create_date(row):
    return pd.Timestamp(year=int(row['year']), month=int(row['quarter'] * 3 - 2), day=1)

df['date'] = df.apply(create_date, axis=1)
year = df.drop_duplicates("year")
year = df

X = df[['realgdp', 'realcons', 'realinv', 'realgovt',
        'realdpi', 'm1', 'tbilrate', 'unemp', 'pop', 'infl', 'realint']].values
# X = df[['realgdp', 'realcons', 'realinv', 'realgovt']].values 
y = df['cpi'].values  # Цільова змінна

y = y.reshape(-1, 1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create model with the same architecture
model = NeuralNetwork(X.shape[1])
model.add_layer(DenseLayer(num_inputs=X.shape[1], num_neurons=8, activation_func=ReLU()))
model.add_layer(DenseLayer(num_inputs=8, num_neurons=1))

# Load pre-trained weights if you have them
model.load_weights('model_weights.pkl')

# Train with gradient descent
history = model.fit(X_scaled, y, 
                   epochs=1000,
                   learning_rate=0.01,
                   verbose=True)

# You can save the weights after training
model.save_weights('new_weights.pkl')