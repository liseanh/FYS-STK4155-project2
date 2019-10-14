import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as sklpre
import sklearn.model_selection as sklms


class RegressionClass:
    def __init__(self, learning_rate=0.1, n_iterations=2000, rtol=0.01, batches=1, penalty=None):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batches = batches
        self.penalty = penalty
        self.rtol = rtol

    def fit(self, X=None, y=None):
        raise RuntimeError("Please do not use this class directly.")

    def gradient_descent(self, beta, X, y):
        for i in range(self.n_iterations):
            grad = self.learning_rate * self.grad_cost_function(beta, X, y)

            rdiff = np.max(np.abs(grad/beta))
            if rdiff < self.rtol:
                print(f"machine broke XD {rdiff}")
                break

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


if __name__ == "__main__":
    pass
