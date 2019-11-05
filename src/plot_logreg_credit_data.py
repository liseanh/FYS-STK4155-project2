import scipy.integrate
import matplotlib.pyplot as plt
import scikitplot as skplt
import numpy as np
import pandas as pd
from main import LogisticRegression

fonts = {
    "font.family": "serif",
    "axes.labelsize": 8,
    "font.size": 8,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}

plt.rcParams.update(fonts)

test_set = np.load("data/credit_data_test.npz")
X_test, y_test = test_set["X_test"], test_set["y_test"].reshape(-1, 1)

model = LogisticRegression()
model.load_model("logreg_credit_model.npz")

y_pred = model.predict_proba(X_test)
proba_0 = 1 - y_pred
proba_1 = y_pred

proba_split = np.append(proba_0, proba_1, axis=1)


def bestCurve(y):
    defaults = np.sum(y == 1, dtype=int)
    total = len(y)
    x = np.linspace(0, 1, total, endpoint=True)
    y1 = np.linspace(0, 1, defaults, endpoint=True)
    y2 = np.ones(total - defaults)
    y3 = np.concatenate([y1, y2])
    return x, y3


x, gains_best = bestCurve(y_test)
fig, ax = plt.subplots()
skplt.metrics.plot_cumulative_gain(y_test.ravel(), proba_split, ax=ax, title=None)
ax.plot(x, gains_best)
ax.legend(["Not default", "Default", "Baseline", "Best model"])
ax.axis([x[0], x[-1], 0, 1.01])
fig.set_size_inches(3.03, 3.03)
fig.tight_layout()
fig.savefig("../doc/figures/cumulative_gain_logreg.pdf", dpi=1000)
fig.clf()


area_baseline = 0.5
area_best = scipy.integrate.simps(gains_best, x) - area_baseline

x, gains_0 = skplt.helpers.cumulative_gain_curve(y_test.ravel(), proba_split[:, 0], 0)
area_0 = scipy.integrate.simps(gains_0, x) - area_baseline

x, gains_1 = skplt.helpers.cumulative_gain_curve(y_test.ravel(), proba_split[:, 1], 1)
area_1 = scipy.integrate.simps(gains_1, x) - area_baseline


ratio_not_default = area_0 / area_best
ratio_default = area_1 / area_best

print(
    f"Area ratio for predicting not default: {ratio_not_default}.\n"
    + f"Area ratio for predicting default: {ratio_default}"
)


df = pd.read_csv("cv_results/results_logreg.csv", header=None, skiprows=1).T

df.columns = df.iloc[0]
df.drop(0, inplace=True)
df["rank_test_score"] = pd.to_numeric(df["rank_test_score"])
df = df.sort_values(by="param_learning_rate", ascending=True)

train_score = df["mean_train_score"].values.astype(np.float)
validation_score = df["mean_test_score"].values.astype(np.float)
learning_rates = df["param_learning_rate"].values.astype(np.float)

fig, ax = plt.subplots()
fig.set_size_inches(3.03, 3.03)
ax.semilogx(learning_rates, train_score, label="train")
ax.semilogx(learning_rates, validation_score, label="validation")
ax.legend()
ax.grid()
ax.set_xlabel(r"Learning rate $\eta$")
ax.set_ylabel("Accuracy")
fig.tight_layout()
fig.savefig("../doc/figures/logreg_learning_rate_accuracy.pdf", dpi=1000)
fig.clf()


# logreg_credit_model
# nn_credit_model
