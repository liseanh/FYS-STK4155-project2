import numpy as np
import joblib
import matplotlib
import matplotlib.pyplot as plt
from main import MultilayerPerceptronRegressor
from mpl_toolkits.mplot3d import Axes3D
import sklearn.preprocessing as sklpre
import sklearn.model_selection as sklms
from main import MultilayerPerceptronRegressor

meshgrid = np.load("data/franke_data_meshgrid.npz")

x_meshgrid, y_meshgrid, z_meshgrid = meshgrid["x"], meshgrid["y"], meshgrid["z"]


training_set = np.load("data/franke_data_train.npz")
test_set = np.load("data/franke_data_test.npz")

X_train, z_train = training_set["X_train"], training_set["z_train"]
X_test, z_test = test_set["X_test"], test_set["z_test"]


scaler = joblib.load("models/franke_data_scaler_features.pkl")
scaler_output = joblib.load("models/franke_data_scaler_output.pkl")

model = MultilayerPerceptronRegressor()
model.load_model("reg_test.npz")

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

y_pred_train = scaler_output.inverse_transform(y_pred_train)
y_pred_test = scaler_output.inverse_transform(y_pred_test)


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
    label="test",
)

ax.axis("off")
ax.grid(False)
ax.set_frame_on(False)
fig.savefig(
    "../doc/figures/3dplot_test.pdf", bbox_inches="tight", pad_inches=0, dpi=1000
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
    label="train",
)


ax.axis("off")
ax.grid(False)
ax.set_frame_on(False)
fig.savefig(
    "../doc/figures/3dplot_train.pdf", bbox_inches="tight", pad_inches=0, dpi=1000
)
# plt.show()
