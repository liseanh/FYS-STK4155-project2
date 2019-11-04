import sys
import numpy as np
from main import MultilayerPerceptronRegressor

try:
    n_x = int(sys.argv[1])
    n_y = int(sys.argv[2])
    sigma = float(sys.argv[3])
except IndexError:
    raise IndexError(
        f"Please input the number of points in x direction, y direction"
        + f" and the standard deviation of the generated data you wish to model"
    )
except ValueError:
    raise TypeError("Input must be integer, integer and float")


training_set = np.load(f"data/franke_data_train_{n_x}_{n_y}_{sigma}.npz")
test_set = np.load(f"data/franke_data_test_{n_x}_{n_y}_{sigma}.npz")

X_train, z_train = training_set["X_train"], training_set["z_train"]
X_test, z_test = test_set["X_test"], test_set["z_test"]

rate = 1e-3
M = "auto"
n = 600

layer_size = [800, 300, 10]

regressor = MultilayerPerceptronRegressor(
    n_epochs=n,
    batch_size=M,
    learning_rate=rate,
    hidden_layer_size=layer_size,
    rtol=-np.inf,
    verbose=True,
    activation_function_output="linear",
)

regressor.fit(X_train, z_train)
print(f"Train R2 score: {regressor.r2_score(X_train, z_train)}")
print(f"Test R2 score: {regressor.r2_score(X_test, z_test)}")

regressor.save_model(f"franke_model_{n_x}_{n_y}_{sigma}.npz")
