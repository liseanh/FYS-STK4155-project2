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
        hidden_layer_size=(20, 10, 5, 3),
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
        # self.backpropagation(X, y)
        # self.feed_forward(X)
        self.gradient_descent(X, y)

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

    def p(self, beta, X):
        exp_expression = np.exp(X @ beta)
        return exp_expression / (1 + exp_expression)

    @staticmethod
    def sigmoid(z):
        """
        The sigmoid function. Use as activation function
        """
        expo = np.exp(z)
        return expo / (1 + expo)

    def activation_function(self, z, layer):
        activation = self.activation_function_list[layer]
        if activation == "sigmoid":
            return self.sigmoid(z)

    @staticmethod
    def softmax(z):
        """
        The softmax function. Can be used as activation function.
        """
        expo = np.exp(z)
        return expo / np.sum(expo, axis=1, keepdims=True)

    def feed_forward(self, X):
        a_i = [X]  # self.activation(X)
        z_i = [0]

        for i in range(self.n_hidden_layers):
            z_i.append(a_i[i] @ self.weights_hidden[i] + self.biases_hidden[i])
            a_i.append(self.sigmoid(z_i[i + 1]))
        z_i.append(a_i[-1] @ self.weights_out + self.biases_out)
        a_i.append(self.sigmoid(z_i[-1]))
        # a_i.pop(0)
        return a_i, z_i

    """
    def backpropagation(self, X, y):
        # output layer
        a_i = self.feed_forward(X)
        # seems like we need to use y.reshape(-1,1) if we don't encode y
        errors = [-y.reshape(-1, 1) + a_i[-1]]
        gradients_weight = [a_i[-2].T @ errors[0]]
        gradients_bias = [np.sum(errors[0], axis=0)]
        # print(gradients_bias[0].shape)
        # second last hidden layer, l = L-1
        errors.append(errors[0] @ self.weights_out.T * a_i[-2] * (1 - a_i[-2]))
        gradients_weight.append(a_i[-3].T @ errors[1])
        gradients_bias.append(np.sum(errors[1], axis=0))
        # print(gradients_bias[1].shape)
        # remaining hidden layers
        for i in range(2, self.n_hidden_layers + 1):
            errors.append(
                errors[i - 1]
                @ self.weights_hidden[-i + 1].T
                * a_i[-i - 1]
                * (1 - a_i[-i - 1])
            )
            gradients_weight.append(a_i[-i - 2].T @ errors[i])
            gradients_bias.append(np.sum(errors[i], axis=0))
            # print(gradients_bias[i].shape)
        # exit()
        return gradients_weight, gradients_bias
        """

    def backpropagation(self, X, y):
        a_i, z_i = self.feed_forward(X)
        delta = np.zeros(self.n_hidden_layers + 1, dtype=np.ndarray)
        gradient_bias = np.zeros_like(delta)
        gradient_weight = np.zeros_like(delta)
        # delta = []
        # gradient_bias = []
        # gradient_weight = []

        # output layer
        # delta.append(self.grad_cost(y, a_i[-1]) * self.grad_activation(z_i[-1]))
        delta[-1] = self.grad_cost(y, a_i[-1]) * self.grad_activation(z_i[-1])
        # print(delta[-1].shape)
        # gradient_bias.append(delta[0])
        gradient_bias[-1] = np.sum(delta[-1], axis=0)
        # print(gradient_bias[-1].shape)
        # gradient_weight.append(delta[0] @ a_i[-2])
        gradient_weight[-1] = delta[-1].T @ a_i[-2]
        # print(gradient_weight[-1].shape)

        # outer hidden layer
        # delta.append(self.weights_out.T @ delta[0] * self.grad_activation(z_i[-2]))
        delta[-2] = self.weights_out @ delta[-1].T * self.grad_activation(z_i[-2]).T
        # print(delta[-2].shape)
        # gradient_bias.append(np.sum(delta[1], axis=0))
        gradient_bias[-2] = np.sum(delta[-2], axis=1)
        # print(gradient_bias[-2].shape)
        # gradient_weight.append(delta[1] * a_i[-3])
        gradient_weight[-2] = delta[-2] @ a_i[-3]
        # print(gradient_weight[-2].shape)
        # print(gradient_bias[-1].shape)
        # print(gradient_bias[-2].shape)
        for l in range(-3, -self.n_hidden_layers - 2, -1):
            """delta.append(
                self.weights_hidden[l + 1].T
                @ delta[l + 1]
                * self.grad_activation(z_i[-(l + 1)])
            )"""
            # print(self.weights_hidden[l + 2].shape)
            # print(self.weights_hidden[l + 2].shape)
            # print(delta[l + 1].shape)
            # print(delta[l + 2].shape)
            delta[l] = (
                self.weights_hidden[l + 2]
                @ delta[l + 1]
                * self.grad_activation(z_i[l]).T
            )
            gradient_bias[l] = np.sum(delta[l], axis=1)
            # gradient_bias.append(delta[l])
            gradient_weight[l] = delta[l] @ a_i[l - 1]
            # gradient_weight.append(delta[l] @ a_i[-(l + 2)])
            # gradient_weight.append(delta[l] @ a_i[l - 1])
            # print(gradient_bias[l].shape)
        return gradient_weight, gradient_bias

    def grad_activation(self, z_i):
        exp_expression = np.exp(-z_i)
        return exp_expression / ((1 + exp_expression) ** 2)

    def grad_cost(self, y, y_pred):
        return y - y_pred

    def gradient_descent(self, X, y):
        n_iterations = len(y) // self.batch_size(len(y))
        y_batches = np.array_split(y, n_iterations)
        X_batches = np.array_split(X, n_iterations, axis=0)
        for i in range(self.n_epochs):
            for j in range(n_iterations):
                random_batch = np.random.randint(n_iterations)
                gradients_weight, gradients_bias = self.backpropagation(
                    X_batches[random_batch], y_batches[random_batch]
                )
                # output layer
                self.weights_out -= self.learning_rate * gradients_weight[0]
                self.biases_out -= self.learning_rate * gradients_bias[0]
                # hidden layer
                for layer in range(1, len(gradients_weight)):
                    # print(f"Grad layer   {layer}", gradients_weight[layer].shape)
                    # print(f"Weight layer {layer}", self.weights_hidden[-layer].shape)
                    # continue
                    self.weights_hidden[-layer] -= (
                        self.learning_rate * gradients_weight[layer]
                    )
                    self.biases_hidden[-layer] -= (
                        self.learning_rate * gradients_bias[layer]
                    )

                # exit()
                """rdiff = np.max(np.abs(grads[-1] / beta[-1]))
                if rdiff < self.rtol:
                    print("Tolerance reached")
                    return

                beta -= grads"""

    def predict(self, X):
        prediction = self.feed_forward(X)[-1]
        print(prediction)
        prediction[prediction >= 0.5] = 1
        prediction[prediction != 1] = 0
        print(prediction)
        return prediction  # .astype(np.int)


if __name__ == "__main__":
    pass
