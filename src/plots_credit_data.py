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

x, y_arrays = bestCurve(y_test)
print(x, y_arrays)

skplt.metrics.plot_cumulative_gain(y_test.ravel(), proba_split)
plt.plot(x, y_arrays)
#skplt.metrics.plot_cumulative_gain(y_test.ravel(), np.random.choice([0, 1], size=[len(y_test.ravel()), 2]))
#skplt.metrics.plot_cumulative_gain(y_test.ravel(), proba_split)
plt.legend(["Not default", "Default", "Baseline", "Perfect model"])
plt.axis([x[0], x[-1], 0, 1.01])
plt.show()
