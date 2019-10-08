import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as sklpre
import sklearn.model_selection as sklms

class RegressionClass:
    def __init__(self, learning_rate=0.1, batches=1, penalty=None):
        self.learning_rate = learning_rate
        self.batches = batches
        self.penalty = penalty

    def fit(self, X=None, y=None):
        raise RuntimeError("Please do not use this class directly.")

    def gradient_descent(self, beta, X, y, N_iterations=1000):
        for i in range(N_iterations):
            beta = -self.learning_rate * self.grad_cost_function(beta, X, y)
        

    def accuracy_score(self, X, y):
        return np.mean(self.predict(X) == y)

    @staticmethod
    def preprocessing(X, y, test_ratio=0.33):
        X = sklpre.scale(X, copy=False)
        X_train, X_test, y_train, y_test = sklms.train_test_split(X, y, test_size=test_ratio)
        return X_train, X_test, y_train, y_test


class LogisticRegression(RegressionClass):
    def fit(self, X, y):
        beta = np.random.normal(0, 1, size=X.shape[1])
        self.gradient_descent(beta, X, y)
        self.beta = beta 

    def p(self, beta, X):
        exp_expression = np.exp(X @ beta)
        return exp_expression / (1 + exp_expression)

    def grad_cost_function(self, beta, X, y):
        return -X.T @ (y - self.p(beta, X))


if __name__ == "__main__":
    pass
