import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from main import MultilayerPerceptronRegressor
import sklearn.preprocessing as sklpre
import sklearn.model_selection as sklms
from mpl_toolkits.mplot3d import Axes3D


training_set = np.load("data/franke_data_train.npz")
test_set = np.load("data/franke_data_test.npz")

X_train, z_train = training_set["X_train"], training_set["z_train"]
X_test, z_test = test_set["X_test"], test_set["z_test"]

rate = 1e-2
M = "auto"  # len(z_train)
n = 1000

layer_size = [18, 12, 6]

regressor = MultilayerPerceptronRegressor(
    n_epochs=n,
    batch_size=M,
    learning_rate=rate,
    hidden_layer_size=layer_size,
    rtol=-np.inf,
    verbose=True,
)

regressor.fit(X_train, z_train)
print(f"Train R2 score: {regressor.r2_score(X_train, z_train)}")
print(f"Test R2 score: {regressor.r2_score(X_test, z_test)}")

regressor.save_model("reg_test.npz")

