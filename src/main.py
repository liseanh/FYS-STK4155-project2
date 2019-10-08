import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as sklpre


class RegressionClass:
    def __init__(self):
        pass

    @staticmethod
    def preprocessing(X, y, test_ratio=0.33):
        X = sklpre.scale(X, copy=False)
        X_train, X_test, y_train, y_test = sklpre.train_test_split(X, y, test_ratio)
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    pass
