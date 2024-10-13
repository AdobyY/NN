class GDOptimizer:
    def __init__(self, eta=0.01):
        self.eta = eta

    def update(self, layer):
        layer.weights -= self.eta * layer.dW
        layer.biases -= self.eta * layer.db
# import numpy as np
# from autograd import grad

# class GDOptimizer():
#     def __init__(self, alpha=0.0, eta=1e-3, **kwargs):
#         self.alpha = alpha
#         self.eta = eta
#         self.score = []
#         self.tol = kwargs.pop("tol", 1e-2)
#         self.v_t = []

#     def update(self, loss, input_tensor, output_tensor, W, **kwargs):
#         verbose = kwargs.pop("verbose", False)
#         to_stop = False
#         loss_grad = grad(loss)
#         if not self.v_t:
#             self.v_t = np.zeros_like(W)
#         self.score.append(loss(W, input_tensor, output_tensor)[0])
#         if verbose:
#             print(f"train score - {self.score[-1]}")
#         grad_W = np.clip(loss_grad(W, input_tensor, output_tensor), -1e6, 1e6)
#         if self.score[-1] <= self.tol:
#             to_stop = True
#         self.v_t = self.alpha * self.v_t + (1.0 - self.alpha) * grad_W
#         W -= self.eta * self.v_t
#         return to_stop, W, self.score[-1]
