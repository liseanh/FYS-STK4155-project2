import numpy as np
from main import MultilayerPerceptronRegressor
import sklearn.preprocessing as sklpre
import sklearn.model_selection as sklms

def generate_Franke_data(x_points, y_points, sigma=0):
    """
    Generates data using Franke's function.
    """
    x_ = np.linspace(0, 1, x_points, endpoint=True)
    y_ = np.linspace(0, 1, y_points, endpoint=True)

    x, y = np.meshgrid(x_, y_)

    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    z = term1 + term2 + term3 + term4 + np.random.normal(0, sigma, size=term1.shape)
    return (x, y, z)


x, y, z = generate_Franke_data(100, 100)

x = x.ravel()
y = y.ravel()
z = z.ravel()

X = np.array([x, y])
print(X.shape)
exit()

rate = 1e-2
M = 200
n = 100

layer_size = [50, 50, 50]

regressor = MultilayerPerceptronRegressor(
    n_epochs=n,
    batch_size=M,
    learning_rate=rate,
    hidden_layer_size=layer_size,
    rtol=1e-5,
)

regressor.fit()
