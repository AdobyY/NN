import numpy as np
from model import NN
from layers import DenseLayer
from optimizers import GDOptimizer 
from losses import MSE
from activations import ReLU

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Створення нейронної мережі
model = NN()
model.add_layer(DenseLayer(8, ReLU()))  # Перший шар з ReLU
model.add_layer(DenseLayer(1, ReLU()))  # Другий шар з ReLU

# Навчання моделі з класичним градієнтним спуском
optimizer = GDOptimizer(eta=0.1)  # Ініціалізація класичного GD
model.fit(X, y, epochs=1000, optimizer=optimizer, loss_fn=MSE)
