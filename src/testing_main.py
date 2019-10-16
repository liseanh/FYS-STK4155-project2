import numpy as np
from main import NeuralNetwork
import sklearn.preprocessing as sklpre
import sklearn.model_selection as sklms

X = np.load("data/design_matrix_credit.npy")
y = np.load("data/targets_credit.npy")

X_train, X_test, y_train, y_test = sklms.train_test_split(X, y, test_size=0.33, stratify=y)


test = NeuralNetwork(n_epochs=200, learning_rate=1e-5, hidden_layer_size=(500,400,300,100))
test.fit(X_train, y_train)
print(test.predict(X_train), y_train)
print(test.accuracy_score(X_train, y_train))
