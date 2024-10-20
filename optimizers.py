import numpy as np

class BaseOptimizer:
    def apply(self, loss, inputs, outputs, weights):
        raise NotImplementedError("Optimizer must implement the apply method")

class SGDOptimizer(BaseOptimizer):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def apply(self, loss, inputs, outputs, weights):
        gradients = loss(inputs, outputs)
        weights -= self.learning_rate * gradients
        return weights

class GeneticAlgorithmOptimizer(BaseOptimizer):
    def __init__(self, population_size=50, mutation_rate=0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = []

    def apply(self, loss, inputs, outputs, weights):
        # Genetic algorithm logic here
        pass  # Implementation left for further development
