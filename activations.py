import numpy as np

class ReLU():
    def activate(self, x):
        return max(0, x)

    def derivative(self, x):
        return np.where(x > 0, 1, 0)

class Sigmoid():
    def activate(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        # Похідна sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        sigmoid_x = 1 / (1 + np.exp(-x))
        return sigmoid_x * (1 - sigmoid_x)

class Tanh():
    def activate(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - np.tanh(x) ** 2