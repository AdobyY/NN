class GDOptimizer:
    def __init__(self, eta=0.01):
        self.eta = eta

    def update(self, layer):
        layer.weights -= self.eta * layer.dW
        layer.biases -= self.eta * layer.db

