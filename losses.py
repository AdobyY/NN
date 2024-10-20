import numpy as np

def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def MSE_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size
