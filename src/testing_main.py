import numpy as np
from main import MultilayerPerceptronClassifier
import sklearn.preprocessing as sklpre
import sklearn.model_selection as sklms
import sklearn.neural_network as sknn

training_set = np.load("data/credit_data_train.npz")
test_set = np.load("data/credit_data_test.npz")

X_train, y_train = training_set["X_train"], training_set["y_train"].reshape(-1, 1)
X_test, y_test = test_set["X_test"], test_set["y_test"].reshape(-1, 1)


rate = 1e-2
M = 200
n = 10000

layer_size = [50, 50, 50, 50]

test = MultilayerPerceptronClassifier(
    n_epochs=n, batch_size=M, learning_rate=rate, hidden_layer_size=layer_size, rtol=1e-5
)

test.fit(X_train, y_train)
print(f"Our: Train accuracy: {test.accuracy_score(X_train, y_train)}. Test accuracy: {test.accuracy_score(X_test, y_test)}")
exit()

reg = sknn.MLPClassifier(
    hidden_layer_sizes=layer_size,
    learning_rate="constant",
    learning_rate_init=rate,
    max_iter=n,
    solver="sgd",
    batch_size=M,
    alpha=0,
    validation_fraction=0,
    momentum=0,
    tol=-np.inf,
    shuffle=False,
    verbose=True,
)

reg.fit(X_train, y_train)
print(f"Scikit: Train accuracy: {reg.score(X_train, y_train)}. Test accuracy: {reg.score(X_test, y_test)}")

reg_test = sknn.MLPClassifier(
    hidden_layer_sizes=layer_size, max_iter=n, verbose=True, tol=-np.inf
)
reg_test.fit(X_train, y_train)
print(f"Scikit: Train accuracy: {reg_test.score(X_train, y_train)}. Test accuracy: {reg_test.score(X_test, y_test)}")
