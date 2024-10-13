class NN:
    def __init__(self, input_size):
        self.layers = []
        self.input_size = input_size  # Зберігаємо вхідний розмір

    def add_layer(self, layer):
        if self.layers:
            layer.initialize(self.layers[-1].n_units)  # Передаємо розмір попереднього шару
        else:
            layer.initialize(self.input_size)  # Вхідний розмір
        self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, y_true, y_pred, loss_fn):
        dA = loss_fn.backward(y_true, y_pred)
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def fit(self, X, y, epochs, optimizer, loss_fn, verbose=False):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = loss_fn.forward(y, y_pred)
            self.backward(y, y_pred, loss_fn)
            for layer in self.layers:
                layer.update(optimizer)
            # if epoch % 100 == 0 and verbose:
            #     print(f'Epoch {epoch}, Loss: {loss}')

