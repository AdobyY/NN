import numpy as np

class MSE:
    @staticmethod
    def forward(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def backward(y_true, y_pred):
        return -2 * (y_true - y_pred) / y_true.size

