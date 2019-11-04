import sys
import numpy as np
import joblib
import matplotlib
import matplotlib.pyplot as plt
from main import MultilayerPerceptronRegressor
from mpl_toolkits.mplot3d import Axes3D
import sklearn.preprocessing as sklpre
import sklearn.model_selection as sklms
from main import MultilayerPerceptronRegressor

try:
    n_x = int(sys.argv[1])
    n_y = int(sys.argv[2])
    sigma = float(sys.argv[3])
except IndexError:
    raise IndexError(
        f"Please input the number of points in x direction, y direction and the "
        + f"standard deviation of the generated data you modelled and wish to plot"
    )
except ValueError:
    raise TypeError("Input must be integer, integer and float")


meshgrid = np.load(f"data/franke_data_meshgrid_{n_x}_{n_y}_{sigma}.npz")

x_meshgrid, y_meshgrid, z_meshgrid = meshgrid["x"], meshgrid["y"], meshgrid["z"]


training_set = np.load(f"data/franke_data_train_{n_x}_{n_y}_{sigma}.npz")
test_set = np.load(f"data/franke_data_test_{n_x}_{n_y}_{sigma}.npz")

X_train, z_train = training_set["X_train"], training_set["z_train"]
X_test, z_test = test_set["X_test"], test_set["z_test"]


scaler = joblib.load(f"models/franke_data_scaler_features_{n_x}_{n_y}_{sigma}.pkl")

model = MultilayerPerceptronRegressor()
model.load_model(f"franke_model_{n_x}_{n_y}_{sigma}.npz")

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)


fig = plt.figure()
fig.set_size_inches(3.03, 1.8)
ax = fig.gca(projection="3d")

surf = ax.plot_surface(
    x_meshgrid,
    y_meshgrid,
    z_meshgrid,
    cmap=matplotlib.cm.coolwarm,
    linewidth=0,
    antialiased=False,
    alpha=0.5,
)

ax.scatter(
    scaler.inverse_transform(X_test)[:, 0],
    scaler.inverse_transform(X_test)[:, 1],
    y_pred_test,
    marker=".",
    s=7,
    label="test",
)

ax.axis("off")
ax.grid(False)
ax.set_frame_on(False)
fig.savefig(
    f"../doc/figures/3dplot_test_{n_x}_{n_y}_{sigma}.pdf",
    bbox_inches="tight",
    pad_inches=0,
    dpi=1000,
)

fig = plt.figure()
fig.set_size_inches(3.03, 1.8)
ax = fig.gca(projection="3d")

surf = ax.plot_surface(
    x_meshgrid,
    y_meshgrid,
    z_meshgrid,
    cmap=matplotlib.cm.coolwarm,
    linewidth=0,
    antialiased=False,
    alpha=0.5,
)

ax.scatter(
    scaler.inverse_transform(X_train)[:, 0],
    scaler.inverse_transform(X_train)[:, 1],
    y_pred_train,
    marker=".",
    s=7,
    label="train",
)

ax.axis("off")
ax.grid(False)
ax.set_frame_on(False)
fig.savefig(
    f"../doc/figures/3dplot_train_{n_x}_{n_y}_{sigma}.pdf",
    bbox_inches="tight",
    pad_inches=0,
    dpi=1000,
)
