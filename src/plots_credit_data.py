import scipy.integrate
import matplotlib.pyplot as plt
import scikitplot as skplt
import numpy as np
from main import MultilayerPerceptronClassifier

test_set = np.load("data/credit_data_test.npz")
X_test, y_test = test_set["X_test"], test_set["y_test"].reshape(-1, 1)

model = MultilayerPerceptronClassifier()
model.load_model("testing.npz")
y_pred = model.predict_proba(X_test)
#print(y_pred.ravel().shape, y_test.shape)
#skplt.metrics.plot_confusion_matrix(y_test.ravel(), y_pred.ravel())
#plt.show()

proba_0 = 1 - y_pred
proba_1 = y_pred

proba_split = np.append(proba_0, proba_1, axis=1)

# proba_and_ideal = np.append(y_test, y_pred, axis=1)
# print(proba_and_ideal.shape)

def bestCurve(y):
	defaults = np.sum(y == 1, dtype=np.int)
	total = len(y)
	x = np.linspace(0, 1, total)
	y1 = np.linspace(0, 1, defaults)
	y2 = np.ones(total-defaults)
	y3 = np.concatenate([y1,y2])
	return x, y3

x, gains_perfect = bestCurve(y_test)
print(x, gains_perfect)


skplt.metrics.plot_cumulative_gain(y_test.ravel(), proba_split)
plt.plot(x, gains_perfect)
#skplt.metrics.plot_cumulative_gain(y_test.ravel(), np.random.choice([0, 1], size=[len(y_test.ravel()), 2]))
#skplt.metrics.plot_cumulative_gain(y_test.ravel(), proba_split)
plt.legend(["Not default", "Default", "Baseline", "Perfect model"])
plt.axis([x[0], x[-1], 0, 1.01])
plt.savefig("../doc/figures/cumulative_gain_NN.pdf", dpi=1000)
plt.close()

area_baseline = 0.5

area_perfect = scipy.integrate.simps(gains_perfect, x) - area_baseline

x, gains_0 = skplt.helpers.cumulative_gain_curve(y_test.ravel(), proba_split[:, 0], 0)
area_0 = scipy.integrate.simps(gains_0, x) - area_baseline

x, gains_1 = skplt.helpers.cumulative_gain_curve(y_test.ravel(), proba_split[:, 1], 1)
area_1 = scipy.integrate.simps(gains_1, x) - area_baseline





ratio_not_default = area_0 / area_perfect
ratio_default = area_1 / area_perfect

print(f"Area ratio for predicting not default: {ratio_not_default}.\nArea ratio for predicting default: {ratio_default}")
