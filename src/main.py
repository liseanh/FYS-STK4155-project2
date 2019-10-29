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
        verbose=False
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

    def fit(self, X=None, y=None):
        raise RuntimeError("Please do not use this class directly.")

    def accuracy_score(self, X, y):
        if len(y.shape) == 1:
            raise ValueError("y-array must have shape (n, 1) Use numpy.reshape(-1, 1)")
        return np.mean(self.predict(X) == np.array(y, dtype=np.int))


class LogisticRegression(RegressionClass):
    def fit(self, X, y):
        beta = np.random.normal(0, np.sqrt(2 / X.shape[1]), size=X.shape[1])
        self.stochastic_gradient_descent(beta, X, y)
        self.beta = beta

    def p(self, beta, X):
        exp_expression = np.exp(X @ beta)
        return exp_expression / (1 + exp_expression)

    def grad_cost_function(self, beta, X, y):
        return -X.T @ (y - self.p(beta, X))

    def stochastic_gradient_descent(self, beta, X, y):
        n_iterations = len(y) // self.batch_size(len(y))
        y_batches = np.array_split(y, n_iterations)
        X_batches = np.array_split(X, n_iterations, axis=0)
        for i in range(self.n_epochs):
            for j in range(n_iterations):
                random_batch = np.random.randint(n_iterations)
                grad = self.learning_rate * self.grad_cost_function(
                    beta, X_batches[random_batch], y_batches[random_batch]
                )
                rdiff = np.max(np.abs(grad / beta))
                if rdiff < self.rtol:
                    print("Tolerance reached")
                    return

                beta -= grad

    def predict(self, X):
        prediction = X @ self.beta
        prediction[prediction >= 0.5] = 1
        prediction[prediction != 1] = 0
        return prediction


class MultilayerPerceptronClassifier(RegressionClass):
    def __init__(
        self,
        hidden_layer_size=(20, 10, 5, 3),
        learning_rate=0.1,
        n_epochs=2000,
        rtol=0.001,
        batch_size="auto",
        penalty=None,
        verbose=False
    ):
        super().__init__(learning_rate, n_epochs, rtol, batch_size, penalty, verbose)
        self.hidden_layer_size = hidden_layer_size
        self.n_hidden_layers = len(hidden_layer_size)

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
        prediction = self.feed_forward(X)[0][-1]
        prediction[prediction >= 0.5] = 1
        prediction[prediction != 1] = 0
        return np.array(prediction, dtype=np.int)

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
        a_i = [X]
        z_i = [0]

        for i in range(self.n_hidden_layers):
            z_i.append(a_i[i] @ self.weights_hidden[i] + self.biases_hidden[i])
            a_i.append(self.sigmoid(z_i[i + 1]))

        z_i.append(a_i[-1] @ self.weights_out + self.biases_out)
        a_i.append(self.sigmoid(z_i[-1]))
        return a_i, z_i

    def backpropagation(self, X, y):
        a_i, z_i = self.feed_forward(X)
        delta = np.zeros(self.n_hidden_layers + 1, dtype=np.ndarray)
        gradient_bias = np.zeros_like(delta)
        gradient_weight = np.zeros_like(delta)

        delta[-1] = self.grad_cost(y, a_i[-1]) * self.grad_activation(z_i[-1])
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
        n_iterations = len(y) // self.batch_size(len(y))
        cost = np.zeros(self.n_epochs)
        y_pred = self.feed_forward(X)[0][-1]
        if self.verbose:
            print(f"INITIAL. Cost func: {self.cost(y,y_pred):g}")

        for i in range(self.n_epochs):
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
                print(f"Epochs {i / self.n_epochs * 100:.2f}% done. Cost func: {cost[i]:g}")
            if i > 10:
                cost_diff = (cost[i - 11 : i] - cost[i - 10 : i + 1]) / cost[i - 11 : i]
                if np.max(cost_diff) < self.rtol:
                    print(
                        f"Loss function did not improve more than given relative tolerance "
                        + f"{self.rtol:g} for 10 consecutive epochs. Stopping at epoch {i:g}"
                    )
                    print(np.max(cost_diff))
                    break

    def activation_function(self, z, layer):
        """
        Not used yet. If we get time, we will modify the class to be able to use
        several different activation functions using this method
        """
        activation = self.activation_function_list[layer]
        if activation == "sigmoid":
            return self.sigmoid(z)

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
    def predict(self, X):
        prediction = self.feed_forward(X)[0][-1]
        return prediction

    @staticmethod
    @numba.njit
    def cost(y, y_pred):
        return np.sum((y_pred - y) ** 2) / 2

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
