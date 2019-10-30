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

skplt.metrics.plot_cumulative_gain(y_test.ravel(), proba_split)
plt.legend(["Not default", "Default"])
plt.show()
