import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sps
import numba


class RegressionClass:
    def __init__(
        self,
        learning_rate=0.1,
        n_epochs=2000,
        rtol=0.01,
        batch_size="auto",
        penalty=None,
        verbose=False,
        learning_schedule=None,
    ):
        if batch_size == "auto":
            self.batch_size = lambda n_inputs: min(200, n_inputs)
        elif batch_size == "none":
            self.batch_size = lambda n_inputs: n_inputs
        elif isinstance(batch_size, int):
            self.batch_size = lambda n_inputs: batch_size
        else:
            raise ValueError("Only 'auto', 'none' or integer supported right now.")
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.penalty = penalty
        self.rtol = rtol
        self.verbose = verbose
        self.learning_schedule = learning_schedule

    def fit(self, X=None, y=None):
        raise RuntimeError("Please do not use this class directly.")

    def accuracy_score(self, X, y):
        if len(y.shape) == 1:
            raise ValueError("y-array must have shape (n, 1) Use numpy.reshape(-1, 1)")
        return np.mean(self.predict(X) == np.array(y, dtype=np.int))


class LogisticRegression(RegressionClass):
    def fit(self, X, y):
        if len(y.shape) == 1:
            raise ValueError("y-array must have shape (n, 1) Use numpy.reshape(-1, 1)")
        self.beta = np.random.normal(0, np.sqrt(2 / X.shape[1]), size=X.shape[1])
        self.stochastic_gradient_descent(X, y)

    def stochastic_gradient_descent(self, X, y):
        if self.learning_schedule == None:
            reduce_i = self.n_epochs + 1
        else:
            reduce_i = self.learning_schedule
        n_iterations = len(y) // self.batch_size(len(y))
        cost = np.zeros(self.n_epochs)
        y_pred = self.predict_proba(X)
        if self.verbose:
            print(f"Initial cost func: {self.cost(y, y_pred):g}")
        for i in range(self.n_epochs):
            if i % reduce_i == 0 and not i == 0:
                self.learning_rate /= 2
                if self.verbose:
                    print(f"Learning rate reduced to {self.learning_rate}")
            batch_indices = np.array_split(np.random.permutation(len(y)), n_iterations)
            for j in range(n_iterations):
                random_batch = np.random.randint(n_iterations)
                gradient = self.grad_cost_function(
                    self.beta,
                    X[batch_indices[random_batch]],
                    y[batch_indices[random_batch]],
                )
                self.beta -= self.learning_rate * gradient
            y_pred = self.predict_proba(X)
            cost[i] = self.cost(y, y_pred)
            if self.verbose:
                print(
                    f"Epochs {i / self.n_epochs * 100:.2f}% done. Cost func: {cost[i]:g}"
                )
            if i > 10:
                cost_diff = (cost[i - 11 : i] - cost[i - 10 : i + 1]) / cost[i - 11 : i]
                if np.max(cost_diff) < self.rtol:
                    print(
                        f"Loss function did not improve more than given relative tolerance "
                        + f"{self.rtol:g} for 10 consecutive epochs. Stopping at epoch {i:g}"
                    )
                    print(np.max(cost_diff))
                    break

    def predict(self, X):
        prediction = self.predict_proba(X)
        prediction[prediction >= 0.5] = 1
        prediction[prediction != 1] = 0
        return prediction

    def predict_proba(self, X):
        exp_expression = np.exp(X @ self.beta)
        exp_expression = exp_expression / (1 + exp_expression)
        return exp_expression.reshape(-1, 1)


    def save_model(self, filename):
        np.savez(f"models/{filename}", beta=self.beta)

    def load_model(self, filename):
        model = np.load(f"models/{filename}", allow_pickle=True)
        self.beta = model["beta"]
 

    @staticmethod
    def cost(y, y_pred):
        return (
            -np.sum(sps.xlogy(y, y_pred) + sps.xlogy(1 - y, 1 - y_pred))
            / y_pred.shape[0]
        )

    @staticmethod
    @numba.njit
    def grad_cost_function(beta, X, y):
        exp_expression = np.exp(X @ beta).reshape(-1, 1)
        exp_expression = exp_expression / (1 + exp_expression)
        return (-X.T @ (y - exp_expression)).sum(axis=1)


class MultilayerPerceptronClassifier(RegressionClass):
    def __init__(
        self,
        hidden_layer_size=(20, 10, 5, 3),
        learning_rate=0.1,
        n_epochs=2000,
        rtol=0.001,
        batch_size="auto",
        penalty=None,
        verbose=False,
        activation_function_output="sigmoid",
        learning_schedule=None,
    ):
        super().__init__(
            learning_rate,
            n_epochs,
            rtol,
            batch_size,
            penalty,
            verbose,
            learning_schedule,
        )
        self.hidden_layer_size = hidden_layer_size
        self.n_hidden_layers = len(hidden_layer_size)
        self.activation_function_output = activation_function_output

    def fit(self, X, y):
        self.n_features = len(X[0, :])
        self.n_inputs = len(X[:, 0])
        if len(y.shape) == 1:
            raise ValueError("y-array must have shape (n, 1) Use numpy.reshape(-1, 1)")
        else:
            self.n_outputs = y.shape[1]

        self.init_biases_weights()
        self.stochastic_gradient_descent(X, y)

    def predict(self, X):
        if self.weights_hidden[0].shape[0] != X.shape[1]:
            print(len(self.weights_hidden[0].shape[0]), X.shape[1])
            raise ValueError(
                "Model was fitted on different inputs than what was provided"
            )

        prediction = self.predict_proba(X)
        prediction[prediction >= 0.5] = 1
        prediction[prediction != 1] = 0
        return np.array(prediction, dtype=np.int)

    def predict_proba(self, X):
        if self.weights_hidden[0].shape[0] != X.shape[1]:
            print(len(self.weights_hidden[0].shape[0]), X.shape[1])
            raise ValueError(
                "Model was fitted on different inputs than what was provided"
            )
        return self.feed_forward(X)[0][-1]

    def accuracy_score(self, X, y):
        if self.weights_out.shape[1] != y.shape[1]:
            print(self.weights_out.shape[1], y.shape[1])
            raise ValueError(
                "Model was fitted on different outputs than what was provided"
            )
        return super().accuracy_score(X, y)

    def save_model(self, filename):
        np.savez(
            f"models/{filename}",
            weights_out=self.weights_out,
            weights_hidden=self.weights_hidden,
            biases_out=self.biases_out,
            biases_hidden=self.biases_hidden,
        )

    def load_model(self, filename):
        model = np.load(f"models/{filename}", allow_pickle=True)
        self.weights_out = model["weights_out"]
        self.weights_hidden = model["weights_hidden"]
        self.biases_out = model["biases_out"]
        self.biases_hidden = model["biases_hidden"]
        self.n_features = self.weights_hidden[0].shape[0]
        self.n_hidden_layers = len(self.weights_hidden)

    def init_biases_weights(self):
        std_weight_init = np.sqrt(1 / self.n_features)

        self.weights_hidden = []
        self.biases_hidden = []

        for i in range(self.n_hidden_layers):
            if i == 0:
                hidden_weights = np.random.uniform(
                    -std_weight_init,
                    std_weight_init,
                    size=(self.n_features, self.hidden_layer_size[i]),
                )
            else:
                hidden_weights = np.random.uniform(
                    -std_weight_init,
                    std_weight_init,
                    size=(self.hidden_layer_size[i - 1], self.hidden_layer_size[i]),
                )

            hidden_biases = np.zeros(self.hidden_layer_size[i]) + 0.01

            self.weights_hidden.append(hidden_weights)
            self.biases_hidden.append(hidden_biases)

        self.weights_out = np.random.normal(
            loc=0,
            scale=std_weight_init,
            size=(self.hidden_layer_size[-1], self.n_outputs),
        )
        self.biases_out = np.zeros(self.n_outputs) + 0.01

    def feed_forward(self, X):
        a_i = np.zeros(self.n_hidden_layers + 2, dtype=np.ndarray)
        z_i = np.zeros(self.n_hidden_layers + 2, dtype=np.ndarray)

        a_i[0] = X
        z_i[0] = np.array([0])

        for i in range(self.n_hidden_layers):

            z_i[i + 1] = a_i[i] @ self.weights_hidden[i] + self.biases_hidden[i]
            a_i[i + 1] = self.sigmoid(z_i[i + 1])

        z_i[-1] = a_i[-2] @ self.weights_out + self.biases_out
        a_i[-1] = self.activation_function_out(z_i[-1], self.activation_function_output)
        return a_i, z_i

    def backpropagation(self, X, y):
        a_i, z_i = self.feed_forward(X)
        delta = np.zeros(self.n_hidden_layers + 1, dtype=np.ndarray)
        gradient_bias = np.zeros_like(delta)
        gradient_weight = np.zeros_like(delta)

        delta[-1] = self.grad_cost(y, a_i[-1]) * self.grad_activation_out(
            z_i[-1], self.activation_function_output
        )
        gradient_bias[-1] = np.sum(delta[-1], axis=0)
        gradient_weight[-1] = (delta[-1].T @ a_i[-2]).T

        delta[-2] = self.weights_out @ delta[-1].T * self.grad_activation(z_i[-2]).T
        gradient_bias[-2] = np.sum(delta[-2], axis=1)
        gradient_weight[-2] = delta[-2] @ a_i[-3]

        for l in range(-3, -self.n_hidden_layers - 2, -1):
            delta[l] = (
                self.weights_hidden[l + 2]
                @ delta[l + 1]
                * self.grad_activation(z_i[l]).T
            )
            gradient_bias[l] = np.sum(delta[l], axis=1)
            gradient_weight[l] = delta[l] @ a_i[l - 1]
        return gradient_weight, gradient_bias

    def stochastic_gradient_descent(self, X, y):
        if self.learning_schedule == None:
            reduce_i = self.n_epochs + 1
        else:
            reduce_i = self.learning_schedule
        n_iterations = len(y) // self.batch_size(len(y))
        cost = np.zeros(self.n_epochs)
        y_pred = self.feed_forward(X)[0][-1]
        if self.verbose:
            print(f"Initial cost func: {self.cost(y,y_pred):g}")

        for i in range(self.n_epochs):
            if i % reduce_i == 0 and not i == 0:
                self.learning_rate /= 2
                if self.verbose:
                    print(f"Learning rate reduced to {self.learning_rate}")
            batch_indices = np.array_split(np.random.permutation(len(y)), n_iterations)
            for j in range(n_iterations):
                random_batch = np.random.randint(n_iterations)
                gradients_weight, gradients_bias = self.backpropagation(
                    X[batch_indices[random_batch]], y[batch_indices[random_batch]]
                )
                # output layer
                self.weights_out -= self.learning_rate * gradients_weight[-1]
                self.biases_out -= self.learning_rate * gradients_bias[-1]
                # hidden layer
                for l in range(-1, -self.n_hidden_layers - 1, -1):
                    self.weights_hidden[l] -= (
                        self.learning_rate * gradients_weight[l - 1].T
                    )
                    self.biases_hidden[l] -= self.learning_rate * gradients_bias[l - 1]
            y_pred = self.feed_forward(X)[0][-1]
            cost[i] = self.cost(y, y_pred)
            if self.verbose:
                print(
                    f"Epochs {i / self.n_epochs * 100:.2f}% done. Cost func: {cost[i]:g}"
                )
            if i > 10:
                cost_diff = (cost[i - 11 : i] - cost[i - 10 : i + 1]) / cost[i - 11 : i]
                if np.max(cost_diff) < self.rtol:
                    print(
                        f"Loss function did not improve more than given relative tolerance "
                        + f"{self.rtol:g} for 10 consecutive epochs. Stopping at epoch {i:g}"
                    )
                    print(np.max(cost_diff))
                    break

    @staticmethod
    @numba.njit
    def activation_function_out(z, activation_function_output):
        """
        Method for ensuring modifyable output activation function. Should expand
        this functionality to all layers if we have the time.
        """
        if activation_function_output == "linear":
            return z
        elif activation_function_output == "sigmoid":
            expo = np.exp(z)
            return expo / (1 + expo)

    @staticmethod
    @numba.njit
    def grad_activation_out(z_i, activation_function_output):
        """
        Method for ensuring modifyable output activation function. Should expand
        this functionality to all layers if we have the time.
        """
        if activation_function_output == "linear":
            return np.ones_like(z_i)
        elif activation_function_output == "sigmoid":
            exp_expression = np.exp(-z_i)
            return exp_expression / ((1 + exp_expression) ** 2)

    @staticmethod
    @numba.njit
    def sigmoid(z):
        """
        The sigmoid function. Use as activation function
        """
        expo = np.exp(z)
        return expo / (1 + expo)

    @staticmethod
    @numba.njit
    def grad_activation(z_i):
        exp_expression = np.exp(-z_i)
        return exp_expression / ((1 + exp_expression) ** 2)

    @staticmethod
    @numba.njit
    def grad_cost(y, y_pred):
        return y_pred - y

    @staticmethod
    def cost(y, y_pred):
        return (
            -np.sum(sps.xlogy(y, y_pred) + sps.xlogy(1 - y, 1 - y_pred))
            / y_pred.shape[0]
        )


class MultilayerPerceptronRegressor(MultilayerPerceptronClassifier):
    def __init__(
        self,
        hidden_layer_size=(20, 10, 5, 3),
        learning_rate=0.1,
        n_epochs=2000,
        rtol=0.001,
        batch_size="auto",
        penalty=None,
        verbose=False,
        activation_function_output="linear",
        learning_schedule=None,
    ):
        super().__init__(
            hidden_layer_size,
            learning_rate,
            n_epochs,
            rtol,
            batch_size,
            penalty,
            verbose,
            activation_function_output,
            learning_schedule,
        )

    def predict(self, X):
        if self.weights_hidden[0].shape[0] != X.shape[1]:
            print(len(self.weights_hidden[0].shape[0]), X.shape[1])
            raise ValueError(
                "Model was fitted on different inputs than what was provided"
            )
        prediction = self.feed_forward(X)[0][-1]
        return prediction

    @staticmethod
    @numba.njit
    def cost(y, y_pred):
        return np.mean((y_pred - y) ** 2) / 2

    @staticmethod
    @numba.njit
    def grad_cost(y, y_pred):
        return y_pred - y

    def accuracy_score(self):
        raise TypeError("Accuracy score is not valid for regression")

    def r2_score(self, X, y):
        return 1 - np.sum((y - self.predict(X)) ** 2) / np.sum((y - np.mean(y)) ** 2)


if __name__ == "__main__":
    pass
