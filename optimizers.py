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
        predictions = network.forward(X) 
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

    def generate_new_population(self, parents):
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
            population = self.generate_new_population(parents)
            if generation % 100 == 0:
                best_fitness = max(fitness_scores)
                print(f"Generation {generation}: Best fitness = {best_fitness}")
        fitness_scores = [self.evaluate_fitness(net, X, y) for net in population]
        best_index = np.argmax(fitness_scores)
        return population[best_index], fitness_scores[best_index]
    
       # Gradient Descent Algorithm
import numpy as np
import copy

class GD:
    def __init__(self, learning_rate=0.01):
        """
        Максимально простий оптимізатор градієнтного спуску
        
        Параметри:
        - learning_rate: швидкість навчання 
        """
        self.learning_rate = learning_rate

    def optimize(self, X, y, network, epochs):
        """
        Оптимізація мережі за допомогою простого градієнтного спуску
        """
        best_loss = float('inf')
        best_network = copy.deepcopy(network)

        for epoch in range(epochs):
            # Пряме поширення
            predictions = network.forward(X)
            
            # Обчислення похибки
            loss = np.mean((predictions - y) ** 2)
            
            # Зворотне поширення та оновлення ваг
            self._backpropagate(network, X, y)
            
            # Оновлення найкращої мережі
            if loss < best_loss:
                best_loss = loss
                best_network = copy.deepcopy(network)
            
            # Виведення інформації про прогрес кожні 100 епох
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss}")

        return best_network, best_loss

    def _backpropagate(self, network, X, y):
        """
        Зворотне поширення для оновлення ваг
        """
        # Обчислення похибки
        predictions = network.forward(X)
        error = predictions - y

        # Зворотне поширення для кожного шару
        for i in range(len(network.layers)-1, -1, -1):
            layer = network.layers[i]
            
            # Обчислення градієнту
            if i == len(network.layers) - 1:
                # Для останнього шару
                output_derivative = error * layer.activation_func.derivative(predictions)
            else:
                # Для проміжних шарів
                next_layer_weights = network.layers[i+1].weights
                next_layer_error = np.dot(output_derivative, next_layer_weights)
                output_derivative = next_layer_error * layer.activation_func.derivative(layer.output)
            
            # Обчислення градієнтів ваг та зміщень
            if i > 0:
                prev_layer_output = network.layers[i-1].output
            else:
                prev_layer_output = X

            # Оновлення ваг
            weight_gradient = np.dot(output_derivative.T, prev_layer_output)
            layer.weights -= self.learning_rate * weight_gradient

            # Оновлення зміщень
            bias_gradient = np.sum(output_derivative, axis=0)
            layer.biases -= self.learning_rate * bias_gradient