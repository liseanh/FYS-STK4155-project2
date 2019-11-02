import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from main import MultilayerPerceptronRegressor
import sklearn.preprocessing as sklpre
import sklearn.model_selection as sklms
from mpl_toolkits.mplot3d import Axes3D


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

exit()

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
    alpha=0.2,
)


y_pred_train = regressor.predict(X_train)
y_pred_test = regressor.predict(X_test)

y_pred_train = scaler_output.inverse_transform(y_pred_train)
y_pred_test = scaler_output.inverse_transform(y_pred_test)

ax.scatter(
    scaler.inverse_transform(X_train)[:, 0],
    scaler.inverse_transform(X_train)[:, 1],
    y_pred_train,
    marker=".",
    label="train",
)
ax.scatter(
    scaler.inverse_transform(X_test)[:, 0],
    scaler.inverse_transform(X_test)[:, 1],
    y_pred_test,
    marker=".",
    label="test",
)
# ax.legend()
ax.axis("off")
ax.grid(False)
ax.set_frame_on(False)
plt.savefig(
    "../doc/figures/regtest.pdf", bbox_inches="tight", pad_inches=0, dpi=1000
)
# plt.show()
