import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as sklpre


class RegressionClass:
    def __init__(self, batches=1, penalty=None):
        self.penalty = penalty
        self.batches = batches

    def fit(self, X=None, y=None):
        raise RuntimeError("Please do not use this class directly.")

    def gradient_descent(self, beta, X, y):
        pass

    def accuracy_score(self, X, y):
        return np.mean(self.predict(X) == y)

    @staticmethod
    def preprocessing(X, y, test_ratio=0.33):
        X = sklpre.scale(X, copy=False)
        X_train, X_test, y_train, y_test = sklpre.train_test_split(X, y, test_ratio)
        return X_train, X_test, y_train, y_test


class LogisticRegression(RegressionClass):
    def fit(self, X, y):
        beta = np.random.normal(0, 1, size=X.shape[0])
        self.gradient_descent(beta, X, y)

    def p(self, beta, X):
        exp_expression = np.exp(X.T @ beta)
        return exp_expression / (1 + exp_expression)

    def grad_cost_function(self, beta, X, y):
        return -X.T @ (y - self.p(beta, X))


if __name__ == "__main__":
    pass
