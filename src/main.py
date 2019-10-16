import numpy as np
import matplotlib.pyplot as plt


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

    def predict(self, X):
        prediction = X @ self.beta
        prediction[prediction >= 0.5] = 1
        prediction[prediction != 1] = 0
        return prediction


class NeuralNetwork(RegressionClass):
    def __init__(
        self,
        hidden_layer_size=(20, 10, 5),
        learning_rate=0.1,
        n_epochs=2000,
        rtol=0.01,
        batch_size="auto",
        penalty=None,
    ):
        super().__init__(learning_rate, n_epochs, rtol, batch_size, penalty)
        self.hidden_layer_size = hidden_layer_size
        self.n_hidden_layers = len(hidden_layer_size)

    def fit(self, X, y):
        self.n_features = len(X[0, :])
        self.n_inputs = len(X[:, 0])
        if len(y.shape) == 1:
            self.n_outputs = 1
        else:
            self.n_outputs = y.shape[1]

        self.init_biases_weights()
        self.backpropagation(X, y)
        # self.feed_forward(X)

    def init_biases_weights(self):
        std_weight_init = np.sqrt(2 / self.n_features)

        self.weights_hidden = []
        self.biases_hidden = []

        for i in range(self.n_hidden_layers):
            if i == 0:
                hidden_weights = np.random.normal(
                    loc=0,
                    scale=std_weight_init,
                    size=(self.n_features, self.hidden_layer_size[i]),
                )
            else:
                hidden_weights = np.random.normal(
                    loc=0,
                    scale=std_weight_init,
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

    @staticmethod
    def sigmoid(X):
        expo = np.exp(X)
        return expo / (1 + expo)

    def feed_forward(self, X):
        a_i = [X]  # self.activation(X)
        for i in range(self.n_hidden_layers):
            a_i.append(
                self.sigmoid(a_i[i] @ self.weights_hidden[i] + self.biases_hidden[i])
            )
        a_i.append(self.sigmoid(a_i[-1] @ self.weights_out + self.biases_out))
        # a_i.pop(0)
        return a_i

    def grad_cost_function(self, model, target):
        return (model - target) / (model * (1 - model))

    def backpropagation(self, X, y):
        a_i = self.feed_forward(X)
        errors = [(y.reshape(-1, 1) - a_i[-1])]
        gradients_weight = [a_i[-2].T @ errors[0]]
        gradients_bias = [np.sum(errors[0], axis=0)]
        print(gradients_bias[0].shape)
        errors.append(errors[0] @ self.weights_out.T * a_i[-2] * (1 - a_i[-1]))
        gradients_weight.append(a_i[-3].T @ errors[1])
        gradients_bias.append(np.sum(errors[1], axis=0))
        print(gradients_bias[1].shape)
        for i in range(2, self.n_hidden_layers + 1):
            errors.append(
                errors[i - 1]
                @ self.weights_hidden[-i + 1].T
                * a_i[-i - 1]
                * (1 - a_i[-i - 1])
            )
            gradients_weight.append(a_i[i - 1].T @ errors[i])
            gradients_bias.append(np.sum(errors[i], axis=0))
            print(gradients_bias[i].shape)
        # exit()
        return gradients_weight, gradients_bias

    def gradient_descent(self, X, y):
        """
        Needs major rework for neural network!
        """
        n_iterations = len(y) // self.batch_size(len(y))
        y_batches = np.array_split(y, n_iterations)
        X_batches = np.array_split(X, n_iterations, axis=0)
        for i in range(self.n_epochs):
            for j in range(n_iterations):
                random_batch = np.random.randint(n_iterations)
                grads = self.learning_rate * self.backpropagation(
                    X_batches[random_batch], y_batches[random_batch]
                )
                rdiff = np.max(np.abs(grads[-1] / beta[-1]))
                if rdiff < self.rtol:
                    print("Tolerance reached")
                    return

                beta -= grads


if __name__ == "__main__":
    pass
