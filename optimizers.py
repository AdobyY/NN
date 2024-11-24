import numpy as np
import copy
from layers import DenseLayer
from model import NeuralNetwork

class GeneticOptimizer:
    def __init__(self, population_size, mutation_rate):
        self.population_size = population_size
        self.mutation_rate = mutation_rate

    def create_network(self, input_size, layers):
        network = NeuralNetwork(input_size)
        for layer in layers:
            network.add_layer(copy.deepcopy(layer))
        return network

    def evaluate_fitness(self, network, X, y):
        predictions = network.forward(X)  # predictions: (batch_size, output_size)
        fitness = -np.mean((predictions - y) ** 2)
        return fitness

    def select_parents(self, population, fitness_scores):
        sorted_indices = np.argsort(fitness_scores)[::-1]
        selected = [population[i] for i in sorted_indices[:len(population)//2]]
        return selected

    def crossover(self, parent1, parent2):
        child = NeuralNetwork(parent1.input_size)
        for layer1, layer2 in zip(parent1.layers, parent2.layers):
            child_layer = DenseLayer(
                num_inputs=layer1.num_inputs,
                num_neurons=layer1.num_neurons,
                activation_func=layer1.activation_func
            )
            # Кросовер ваг
            mask = np.random.rand(*layer1.weights.shape) > 0.5
            child_layer.weights = np.where(mask, layer1.weights, layer2.weights)
            # Кросовер зміщень
            mask_b = np.random.rand(*layer1.biases.shape) > 0.5
            child_layer.biases = np.where(mask_b, layer1.biases, layer2.biases)
            child.layers.append(child_layer)
        return child

    def mutate(self, network):
        for layer in network.layers:
            layer.mutate(self.mutation_rate)

    def generate_new_population(self, parents, input_size, layers):
        new_population = parents.copy()
        while len(new_population) < self.population_size:
            parent_indices = np.random.choice(len(parents), 2, replace=False) #
            parent1, parent2 = parents[parent_indices[0]], parents[parent_indices[1]]
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_population.append(child)
        return new_population

    def optimize(self, X, y, network, generations):
        population = [self.create_network(network.input_size, network.layers) for _ in range(self.population_size)]
        for generation in range(generations):
            fitness_scores = [self.evaluate_fitness(net, X, y) for net in population]
            parents = self.select_parents(population, fitness_scores)
            population = self.generate_new_population(parents, network.input_size, network.layers)
            if generation % 100 == 0:
                best_fitness = max(fitness_scores)
                print(f"Generation {generation}: Best fitness = {best_fitness}")
        fitness_scores = [self.evaluate_fitness(net, X, y) for net in population]
        best_index = np.argmax(fitness_scores)
        return population[best_index], fitness_scores[best_index]