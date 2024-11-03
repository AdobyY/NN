import numpy as np

class ReLU:
    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x):
        return (x > 0).astype(float)
    
class Sigmoid:
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x):
        sig = self.forward(x)
        return sig * (1 - sig)

