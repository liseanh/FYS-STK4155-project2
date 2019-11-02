import numpy as np
import sklearn.model_selection as sklms
import sklearn.preprocessing as sklpre


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


x, y, z = generate_Franke_data(200, 200, 0.1)

x_meshgrid = x.copy()
y_meshgrid = y.copy()
z_meshgrid = z.copy()

x = x.ravel()
y = y.ravel()
z = z.ravel().reshape(-1, 1)

X = np.array([x, y]).T

X_train, X_test, z_train, z_test = sklms.train_test_split(X, z, test_size=0.33)

scaler_output = sklpre.MinMaxScaler().fit(z_train)

z_train = scaler_output.transform(z_train)
z_test = scaler_output.transform(z_test)


scaler = sklpre.StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

np.savez("data/franke_data_train.npz", X_train=X_train, z_train=z_train)
np.savez("data/franke_data_test.npz", X_test=X_test, z_test=z_test)
np.savez("data/franke_data_meshgrid.npz", x=x_meshgrid, y=y_meshgrid, z=z_meshgrid)