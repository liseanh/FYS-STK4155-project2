import numpy as np
from main import NeuralNetwork
import sklearn.preprocessing as sklpre
import sklearn.model_selection as sklms
import sklearn.neural_network as sknn

training_set = np.load("data/credit_data_train.npz")
test_set = np.load("data/credit_data_test.npz")

X_train, y_train = training_set["X_train"], training_set["y_train"]
X_test, y_test = test_set["X_test"], test_set["y_test"]

rate = 0.1
M = 80
n = 500

layer_size = (50,50,50,50)


test = NeuralNetwork(
    n_epochs=n, batch_size=M, learning_rate=rate, hidden_layer_size=layer_size, rtol=1e-5
)

test.fit(X_train, y_train)
#test.predict(X_train)
print(test.accuracy_score(X_train, y_train))

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
    tol=0,
    shuffle=False,
    verbose=True,
)

reg = reg.fit(X_train, y_train)
pred = reg.predict(X_test)
print(reg.score(X_test, y_test), pred)

exit()
reg_test = sknn.MLPClassifier(
    hidden_layer_sizes=layer_size, activation="relu", max_iter=10000, verbose=True
)
reg_test = reg_test.fit(X_train, y_train)
pred = reg_test.predict(X_test)
print(reg_test.score(X_test, y_test))
print(reg_test.score(X_train, y_train))
