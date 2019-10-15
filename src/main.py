import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as sklpre
import sklearn.model_selection as sklms


class RegressionClass:
    def __init__(
        self,
        learning_rate=0.1,
        n_epochs=2000,
        rtol=0.01,
        batch_size="auto",
        penalty=None,
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

    def fit(self, X=None, y=None):
        raise RuntimeError("Please do not use this class directly.")

    def gradient_descent(self, beta, X, y):
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

    def accuracy_score(self, X, y):
        return np.mean(self.predict(X) == y)


class LogisticRegression(RegressionClass):
    def fit(self, X, y):
        beta = np.random.normal(0, np.sqrt(2 / X.shape[1]), size=X.shape[1])
        self.gradient_descent(beta, X, y)
        self.beta = beta

    def p(self, beta, X):
        exp_expression = np.exp(X @ beta)
        return exp_expression / (1 + exp_expression)

    def grad_cost_function(self, beta, X, y):
        return -X.T @ (y - self.p(beta, X))

    def predict(self, X):
        prediction = X @ self.beta
        prediction[prediction >= 0.5] = 1
        prediction[prediction != 1] = 0
        return prediction


class NeuralNetwork(RegressionClass):
    def __init__(self, hidden_layer_size=(30, 10, 10)):
        super.__init__(
            self,
            learning_rate=0.1,
            n_epochs=2000,
            rtol=0.01,
            batch_size="auto",
            penalty=None,
        )
        self.hidden_layer_size = hidden_layer_size
        self.n_hidden_layers = len(hidden_layer_size)

    def fit(X, y):
        self.n_features = len(X[0, :])
        self.n_inputs = len(X[:, 0])
        if len(y.shape) == 1:
            self.n_outputs = 1
        else:
            self.n_outputs = y.shape[1]

    def init_biases_weights():
        self.weights_biases_hidden = []
        for i in range(self.n_hidden_layers):
            if i == 0:
                weights_and_biases = np.zeros(
                    (self.n_inputs + 1, self.hidden_layer_size[i])
                )

            else:
                weights_and_biases = np.zeros(
                    (self.hidden_layer_size[i - 1], self.hidden_layer_size[i])
                )
            weights_and_biases[:, 0] = 0.01
            weights_and_biases[:, 1:] = np.random.normal(
                0,
                scale=np.sqrt(2 / self.n_features),
                size=weights_and_biases[:, 1:].shape,
            )
            self.weights_biases_hidden.append(weights_and_biases)

        self.weights_biases_output = np.zeros(
            (self.hidden_layer_size[-1], self.n_outputs)
        )
        self.weights_and_biases_output[:, 0] = 0.01
        self.weights_and_biases_output[:, 1:] = np.random.normal(
            0,
            scale=np.sqrt(2 / self.n_features),
            size=self.weights_and_biases_output[:, 1:].shape,
        )


if __name__ == "__main__":
    pass
