import sys
import numpy as np
import joblib
import matplotlib
import matplotlib.cm as cm
import pandas as pd
import matplotlib.pyplot as plt
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

fonts = {
    "font.family": "serif",
    "axes.labelsize": 8,
    "font.size": 8,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}

plt.rcParams.update(fonts)

meshgrid = np.load(f"data/franke_data_meshgrid_{n_x}_{n_y}_{sigma}.npz")

x_meshgrid, y_meshgrid, z_meshgrid = meshgrid["x"], meshgrid["y"], meshgrid["z"]


training_set = np.load(f"data/franke_data_train_{n_x}_{n_y}_{sigma}.npz")
test_set = np.load(f"data/franke_data_test_{n_x}_{n_y}_{sigma}.npz")

X_train, z_train = training_set["X_train"], training_set["z_train"].reshape(-1, 1)
X_test, z_test = test_set["X_test"], test_set["z_test"].reshape(-1, 1)


scaler = joblib.load(f"models/franke_data_scaler_features_{n_x}_{n_y}_{sigma}.pkl")

model = MultilayerPerceptronRegressor()
model.load_model(f"franke_model_{n_x}_{n_y}_{sigma}.npz")

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)


fig = plt.figure()
fig.set_size_inches(3.03, 3.03)
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
fig.set_size_inches(3.03, 3.03)
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

# Plotting hyperparameter search
df = pd.read_csv(
    f"cv_results/results_nn_franke_{n_x}_{n_y}_{sigma}.csv", header=None, skiprows=1
).T

df.columns = df.iloc[0]
df.drop(0, inplace=True)
df["rank_test_score"] = pd.to_numeric(df["rank_test_score"])
df = df.sort_values(by="param_learning_rate", ascending=True)

train_score = df["mean_train_score"].values.astype(np.float)
validation_score = df["mean_test_score"].values.astype(np.float)
learning_rates = df["param_learning_rate"].values.astype(np.float)
lambdas = df["param_lambd"].values.astype(np.float)
fig, ax = plt.subplots()
fig.set_size_inches(3.03, 3.03)
ax.scatter(learning_rates, lambdas, c=validation_score, s=20, cmap=cm.coolwarm)
ax.set_xlabel("Learning rate")
ax.set_ylabel(r"$\lambda$")
ax.set_xlim([0, np.max(learning_rates)*1.1])
cbar = fig.colorbar(
    cm.ScalarMappable(
        norm=cm.colors.Normalize(
            vmin=validation_score.min(), vmax=validation_score.max()
        ),
        cmap=cm.coolwarm,
    ),
    ax=ax,
)
cbar.set_label(r"R$^2$ score")
fig.tight_layout()
fig.savefig(
    f"../doc/figures/nn_learning_rate_lambda_r2_franke_{n_x}_{n_y}_{sigma}.pdf",
    dpi=1000,
)
fig.clf()


print(
    f"R2 score test: {model.r2_score(X_test, z_test)}.\n"
    + f"R2 score train: {model.r2_score(X_train, z_train)}"
)
