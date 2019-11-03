import sys
import numpy as np
import joblib
import sklearn.model_selection as sklms
import sklearn.preprocessing as sklpre


def generate_Franke_data(x_points, y_points, std=0):
    """
    Generates data using Franke's function with added Gaussian noise around 0.

    Parameters:

    x_points: int
        Number of points in the x-direction

    y_points : int 
        Number of points in the y-direction

    std: int, default=0 
        Standard deviation of the added Gaussian noise

    """
    
    x_ = np.linspace(0, 1, x_points, endpoint=True)
    y_ = np.linspace(0, 1, y_points, endpoint=True)

    x, y = np.meshgrid(x_, y_)

    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    z = term1 + term2 + term3 + term4 + np.random.normal(0, std, size=term1.shape)
    return (x, y, z)


try:
    n_x = int(sys.argv[1])
    n_y = int(sys.argv[2])
    sigma = float(sys.argv[3])
except IndexError:
    raise IndexError(
        f"Please input the number of points in x direction, y direction"
        + f" and the standard deviation"
    )
except ValueError:
    raise TypeError("Input must be integer, integer and float")


x, y, z = generate_Franke_data(n_x, n_y, sigma)

x_meshgrid = x.copy()
y_meshgrid = y.copy()
z_meshgrid = z.copy()

x = x.ravel()
y = y.ravel()
z = z.ravel().reshape(-1, 1)

X = np.array([x, y]).T

X_train, X_test, z_train, z_test = sklms.train_test_split(X, z, test_size=0.33)


scaler = sklpre.StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, f"models/franke_data_scaler_features_{n_x}_{n_y}_{sigma}.pkl")
np.savez(f"data/franke_data_train_{n_x}_{n_y}_{sigma}.npz", X_train=X_train, z_train=z_train)
np.savez(f"data/franke_data_test_{n_x}_{n_y}_{sigma}.npz", X_test=X_test, z_test=z_test)
np.savez(f"data/franke_data_meshgrid_{n_x}_{n_y}_{sigma}.npz", x=x_meshgrid, y=y_meshgrid, z=z_meshgrid)
