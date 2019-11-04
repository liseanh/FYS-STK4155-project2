import numpy as np
from main import LogisticRegression
import sklearn.preprocessing as sklpre
import sklearn.model_selection as sklms
import sklearn.neural_network as sknn

training_set = np.load("data/credit_data_train.npz")
test_set = np.load("data/credit_data_test.npz")

X_train, y_train = training_set["X_train"], training_set["y_train"].reshape(-1, 1)
X_test, y_test = test_set["X_test"], test_set["y_test"].reshape(-1, 1)

reg = LogisticRegression(learning_rate=1e-3, verbose=True)
reg.fit(X_train, y_train)
print(reg.accuracy_score(X_test, y_test))
