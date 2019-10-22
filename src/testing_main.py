import numpy as np
from main import NeuralNetwork
import sklearn.preprocessing as sklpre
import sklearn.model_selection as sklms
import sklearn.neural_network as sknn

X = np.load("data/design_matrix_credit.npy")
y = np.load("data/targets_credit.npy")
X_train, X_test, y_train, y_test = sklms.train_test_split(
    X, y, test_size=0.33, stratify=y
)

rate = 1
M = 200
n = 2000

layer_size = (18, 12, 6)

test = NeuralNetwork(
    n_epochs=n, batch_size=M, learning_rate=rate, hidden_layer_size=layer_size
)

test.fit(X_train, y_train)
test.predict(X_train)
print(test.accuracy_score(X_train, y_train))
exit()

reg = sknn.MLPClassifier(
    hidden_layer_sizes=layer_size,
    activation="logistic",
    learning_rate="constant",
    learning_rate_init=rate,
    max_iter=n,
    solver="sgd",
    batch_size=M,
    alpha=0,
    validation_fraction=0,
    momentum=0,
    tol=0,
    shuffle=False,
    verbose=True,
)

reg = reg.fit(X_train, y_train)
pred = reg.predict(X_test)
print(reg.score(X_test, y_test), pred)

reg_test = sknn.MLPClassifier(
    hidden_layer_sizes=layer_size, activation="relu", max_iter=10000, verbose=True
)
reg_test = reg_test.fit(X_train, y_train)
pred = reg_test.predict(X_test)
print(reg_test.score(X_test, y_test))
print(reg_test.score(X_train, y_train))
