def ReLU(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

class Dense(BaseLayer):
    """
    Dense Layer
    -----------
    The essential building block of an ANN.
    """
    def init(
        self, size: int, nonlinearity: tp.Callable[[tp.Any], np.ndarray], **kwargs
    ):
        self.size = size
        super().init(nonlinearity=nonlinearity, **kwargs)

    def build_weights_dict(self, input_shape):
        input_size = np.prod(input_shape, dtype=int)
        self.parser.add_weights("params", (input_size, self.size))
        self.parser.add_weights("biases", (self.size,))
        return self.parser.N, (self.size,)

    def forward(self, inputs, param_vector):
        params = self.parser.get(param_vector, "params")
        biases = self.parser.get(param_vector, "biases")
        if inputs.ndim > 2:
            inputs = inputs.reshape((inputs.shape[0], np.prod(inputs.shape[1:])))
        return self.nonlinearity(np.dot(inputs[:, :], params) + biases)

def MSE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Loss"""
    return np.mean((y_true - y_pred) ** 2, 0)

class FFN(object):
    """
    Feed Forward Network
    --------------------
    """

    def init(
        self,
        input_shape: tp.Tuple[int],
        layer_specs: tp.List[BaseLayer],
        loss: tp.Callable[..., np.ndarray],
        **kwargs,
    ) -> None:

        self.parser = WeightsParser()
        self.regularization = kwargs.get("regularization", "l2")
        self.reg_coef = kwargs.get("reg_coef", 0)
        self.layer_specs = layer_specs
        cur_shape = input_shape
        W_vect = np.array([])
        for num, layer in enumerate(self.layer_specs):
            layer.number = num
            N_weights, cur_shape = layer.build_weights_dict(cur_shape)
            self.parser.add_weights(str(layer), (N_weights,))
            W_vect = np.append(W_vect, layer.initializer(size=(N_weights,)))
        self._loss = loss
        self.W_vect = 0.1 * W_vect

    def loss(
        self, W_vect: np.ndarray, X: np.ndarray, y: np.ndarray, omit_reg: bool = False
    ) -> np.ndarray:

        if self.regularization == "l2" and not omit_reg:
            reg = np.power(np.linalg.norm(W_vect, 2), 2)
        elif self.regularization == "l1" and not omit_reg:
            reg = np.linalg.norm(W_vect, 1)
        else:
            reg = 0.0
        return self._loss(self._predict(W_vect, X), y) + self.reg_coef * reg

    def predict(self, inputs: np.ndarray) -> np.ndarray:

        return self._predict(self.W_vect, inputs)

    def _predict(self, W_vect: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        cur_units = inputs
        for layer in self.layer_specs:
            cur_weights = self.parser.get(W_vect, str(layer))
            cur_units = layer.forward(cur_units, cur_weights)
        return cur_units

    def eval(self, input: np.ndarray, output: np.ndarray) -> float:
        return self.loss(self.W_vect, input, output, omit_reg=True)

    def frac_err(self, X, T):
        return np.mean(
            np.argmax(T, axis=1) != np.argmax(self.predict(self.W_vect, X), axis=1)
        )

    def fit(
        self,
        optimiser: BaseOptimizer,
        train_sample: tp.Tuple[np.ndarray],
        validation_sample: tp.Tuple[np.ndarray],
        batch_size: int,
        epochs: tp.Optional[int] = None,
        verbose: tp.Optional[bool] = None,
        load_best_model_on_end: bool = True,
        minimize_metric: bool = True,
    ):

        self._optimiser = optimiser

        verbose = verbose if verbose else False
        epochs = epochs if epochs else 1

        inst = None
        best_inst = None
        best_score = np.inf if minimize_metric else -np.inf
        best_epoch = 0

        history = dict(epoch=[], train_loss=[], validation_loss=[])

        for i in tqdm(range(epochs), desc="Training "):

            tr_accum_loss = []
            tr_loss = np.inf
            to_stop = False


class SGDOptimizer(BaseOptimizer):
    """
    Stochastic Gradient Descent Optimizer
    """

    def __init__(self, alpha: float = 0.0, eta: float = 1e-3, **kwargs):

        self._alpha = alpha
        self._eta = eta
        self._score = []
        self._tol = kwargs.pop("tol", 1e-2)
        self._v_t = []
        super().__init__(**kwargs)

    def apply(
        self,
        loss: tp.Callable[
            [np.ndarray, np.ndarray, np.ndarray, tp.Optional[tp.Dict[str, float]]],
            float,
        ],
        input_tensor: np.ndarray,
        output_tensor: np.ndarray,
        W: np.ndarray,
        **kwargs,
    ):

        verbose = kwargs.pop("verbose", False)
        to_stop = False
        loss_grad = grad(loss)
        if not (len(self._v_t)):
            self._v_t = np.zeros_like(W)
        self._score.append(loss(W, input_tensor, output_tensor)[0])
        if verbose:
            print(f"train score - {self._score[-1]}")
        grad_W = np.clip(
            loss_grad(W, input_tensor, output_tensor),
            -1e6,
            1e6,
        )
        if self._score[-1] <= self._tol:
            to_stop = True
        self._v_t = self._alpha * self._v_t + (1.0 - self._alpha) * grad_W
        W -= self._eta * self._v_t
        return to_stop, W, self._score[-1]







class GeneticAlgorithmOptimizer(BaseOptimizer):
    """
    `Vanilla` Genetic Algorithm.
    """

    def __init__(self, num_population: int, k: int = 5, **kwargs):
   
        self._num_population = num_population
        self._k = k
        self._population = None
        self._best_iter = None
        self._last_score = np.inf
        self._iter = 0
        self._tol = kwargs.pop("tol", 1e-2)
        super().__init__(**kwargs)

    @staticmethod
    def construct_genome(W: np.ndarray, weight_init: tp.Callable[..., np.ndarray]):
   
        return 0.1 * weight_init(0, 1, size=W.shape)

    @staticmethod
    def crossover(ind_1: np.ndarray, ind_2: np.ndarray) -> np.ndarray:

        assert len(ind_1) == len(ind_2), "individuals must have same len"
        index = np.random.default_rng().integers(len(ind_1))
        ind_12 = np.concatenate((ind_1[:index], ind_2[index:]), axis=None)
        ind_21 = np.concatenate((ind_2[:index], ind_1[index:]), axis=None)
        return ind_12, ind_21

    @staticmethod
    def mutate(
        ind: np.ndarray, mu: float = 0.1, sigma: float = 1.0, factor: float = 0.01
    ) -> tp.List[BaseLayer]:
  
        seed = int(datetime.utcnow().timestamp() * 1e5)
        ind += factor * np.random.default_rng(seed).normal(
            loc=mu, scale=sigma, size=len(ind)
        )
        return ind

    def apply(
        self,
        loss: tp.Callable[
            [np.ndarray, np.ndarray, tp.Optional[tp.Dict[str, float]]], float
        ],
        input_tensor: np.ndarray,
        output_tensor: np.ndarray,
        W: np.ndarray,
        **kwargs,
    ):
 
        verbose = kwargs.pop("verbose", False)
        seed = int(datetime.utcnow().timestamp() * 1e5)
        to_stop = False
        if not (self._population):
            population = [
                self.construct_genome(W, np.random.default_rng(seed + 42 * i).normal)
                for i in range(self._num_population)
            ]
        else:
            population = self._population[:]
        scores = []
        for g in population:
            scores.append(loss(g, input_tensor, output_tensor)[0])
        scores, scores_idx = np.sort(scores), np.argsort(scores)
        if verbose:
            print(f"best individual - {scores[0]}")

        if scores[0] < self._tol:
            to_stop = True
        self._population = np.array(population)[scores_idx][
            : self._num_population - self._k * 3
        ].tolist()
        probas = 1.0 - (scores - np.min(scores)) / np.ptp(scores)
        probas /= sum(probas)
        for _ in range(self._k):
            indices = np.random.default_rng(seed).choice(scores_idx, 2, p=probas)
            ind_1, ind_2 = self.crossover(
                population[indices[0]], population[indices[1]]
            )
            self._population.append(
                self.mutate(ind_1, factor=np.random.default_rng(seed).normal(0.01, 0.1))
            )
            self._population.append(
                self.mutate(ind_2, factor=np.random.default_rng(seed).normal(0.01, 0.1))
            )
        idx_survived = np.random.default_rng().choice(
            scores_idx[: len(population)], self._k
        )
        for idx in idx_survived:
            self._population.append(
                self.mutate(
                    population[idx], factor=np.random.default_rng(seed).normal(0.1, 0.1)
                )
            )
        self._iter += 1
        return to_stop, np.array(population[scores_idx[0]]), scores[0]
    
    