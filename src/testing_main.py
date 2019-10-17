import numpy as np
from main import NeuralNetwork
import sklearn.preprocessing as sklpre
import sklearn.model_selection as sklms
import sklearn.neural_network as sknn

X = np.load("data/design_matrix_credit.npy")
y = np.load("data/targets_credit.npy")

X_train, X_test, y_train, y_test = sklms.train_test_split(X, y, test_size=0.33, stratify=y)


layer_size = (1000, 800,500)

test = NeuralNetwork(
    n_epochs=300, batch_size=3000, learning_rate=1e-5, hidden_layer_size=layer_size
)

test.fit(X_train, y_train)
test.predict(X_train)
print(test.accuracy_score(X_train, y_train))
# print(layer_size)

reg = sknn.MLPClassifier(
    hidden_layer_sizes=layer_size,
    activation="logistic",
    learning_rate="constant",
    learning_rate_init=1e-5,
    max_iter=300,
    solver="sgd",
    batch_size=3000,
    tol=0,
    alpha=0,
    validation_fraction=0,
)

reg = reg.fit(X_train, y_train)
pred = reg.predict(X_test)
print(reg.score(X_test, y_test))
