import numpy as np
from main import MultilayerPerceptronRegressor

training_set = np.load("data/franke_data_train.npz")
test_set = np.load("data/franke_data_test.npz")

X_train, z_train = training_set["X_train"], training_set["z_train"]
X_test, z_test = test_set["X_test"], test_set["z_test"]

rate = 1e-3 * 2
M = "auto"  # len(z_train)
n = 500

layer_size = [50, 25, 5]

regressor = MultilayerPerceptronRegressor(
    n_epochs=n,
    batch_size=M,
    learning_rate=rate,
    hidden_layer_size=layer_size,
    rtol=-np.inf,
    verbose=True,
    activation_function_output="linear"
)

regressor.fit(X_train, z_train)
print(f"Train R2 score: {regressor.r2_score(X_train, z_train)}")
print(f"Test R2 score: {regressor.r2_score(X_test, z_test)}")

regressor.save_model("reg_test.npz")
