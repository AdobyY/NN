class GDOptimizer:
    def __init__(self, eta=0.01):
        self.eta = eta

    def update(self, layer):
        layer.weights -= self.eta * layer.dW
        layer.biases -= self.eta * layer.db

import numpy as np

class GeneticAlgorithmOptimizer:
    def __init__(self, population_size=50, mutation_rate=0.01, crossover_rate=0.7):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def initialize_population(self, layer):
        population = []
        for _ in range(self.population_size):
            individual = {
                'weights': layer.weights.copy(),
                'biases': layer.biases.copy()
            }
            population.append(individual)
        return population

    def select_parents(self, population, fitnesses, num_parents):
        parents_idx = np.argsort(fitnesses)[-num_parents:]
        return [population[i] for i in parents_idx]

    def crossover(self, parent1, parent2):
        child = {}
        child['weights'] = (parent1['weights'] + parent2['weights']) / 2
        child['biases'] = (parent1['biases'] + parent2['biases']) / 2
        return child

    def mutate(self, individual):
        if np.random.rand() < self.mutation_rate:
            individual['weights'] += np.random.randn(*individual['weights'].shape) * 0.1
        if np.random.rand() < self.mutation_rate:
            individual['biases'] += np.random.randn(*individual['biases'].shape) * 0.1

    def update(self, layers, X, y):
        for layer in layers:
            population = self.initialize_population(layer)
            fitnesses = np.array([self.fitness(ind, layer, X, y) for ind in population])
            parents = self.select_parents(population, fitnesses, self.population_size // 2)
            
            new_population = []
            while len(new_population) < self.population_size:
                parent1, parent2 = np.random.choice(parents, 2, replace=False)
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                new_population.append(child)
            
            best_individual = population[np.argmax(fitnesses)]
            layer.set_weights(best_individual['weights'])
            layer.set_biases(best_individual['biases'])

    def fitness(self, individual, layer, X, y):
        layer.set_weights(individual['weights'])
        layer.set_biases(individual['biases'])
        output = layer.forward(X)
        loss = np.mean((y - output) ** 2)
        return 1 / (loss + 1e-6)

    def forward(self, X, weights, biases):
        z = np.dot(X, weights) + biases
        return 1 / (1 + np.exp(-z))
