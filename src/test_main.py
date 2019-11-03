import numpy as np
from main import MultilayerPerceptronClassifier, MultilayerPerceptronRegressor
import sklearn.preprocessing as sklpre
import matplotlib.pyplot as plt

np.random.seed(len("jeff"))


def test_MultilayerPerceptronRegressor_overfit_simple():
    x = np.linspace(0, 5, 10000, endpoint=True)
    y = 3 * x
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    scalerx = sklpre.StandardScaler().fit(x)
    x = scalerx.transform(x)
    model = MultilayerPerceptronRegressor(
        n_epochs=100,
        hidden_layer_size=[3],
        rtol=-np.inf,
        learning_rate=2e-3,
        verbose=False,
    )
    model.fit(x, y)
    y_model = model.predict(x)
    np.testing.assert_array_almost_equal(y, y_model, decimal=1)


def test_MultilayerPerceptronClassifier_overfit_simple():
    y = np.zeros(1000)
    X = np.random.randint(0, 2, size=y.shape)
    y[X == 1] = 1
    X = X.reshape(-1, 1)
    y = y.reshape(-1, 1)
    model = MultilayerPerceptronClassifier(
        n_epochs=50,
        batch_size=32,
        hidden_layer_size=[10],
        learning_rate=1,
        verbose=False,
    )
    model.fit(X, y)
    y_model = model.predict(X)
    np.testing.assert_array_equal(y, y_model)
