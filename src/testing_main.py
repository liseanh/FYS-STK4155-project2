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

rate = 1e-5
M = 200
n = 2000

layer_size = (18, 12, 6)

test = NeuralNetwork(
    n_epochs=n, batch_size=M, learning_rate=rate, hidden_layer_size=layer_size
)

test.fit(X_train, y_train)
test.predict(X_train)
print(test.accuracy_score(X_train, y_train))
# print(layer_size)

reg = sknn.MLPClassifier(
    hidden_layer_sizes=layer_size,
    activation="logistic",
    learning_rate="constant",
    learning_rate_init=rate,
    max_iter=n,
    solver="sgd",
    batch_size=M,
    tol=0,
    alpha=0,
    validation_fraction=0,
    momentum=0,
    nesterovs_momentum=False,
    shuffle=False,
)

reg = reg.fit(X_train, y_train)
pred = reg.predict(X_test)
print(reg.score(X_test, y_test), pred)
