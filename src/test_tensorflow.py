import tensorflow as tf
import numpy as np


training_set = np.load("data/credit_data_train.npz")
test_set = np.load("data/credit_data_test.npz")

X_train, y_train = training_set["X_train"], training_set["y_train"].reshape(-1, 1)
X_test, y_test = test_set["X_test"], test_set["y_test"].reshape(-1, 1)


@tf.function
def leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.01)


model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(50, activation=leaky_relu),
        tf.keras.layers.Dense(50, activation=leaky_relu),
        tf.keras.layers.Dense(50, activation=leaky_relu),
        tf.keras.layers.Dense(50, activation=leaky_relu),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)
# Compile model
model.compile(
    loss="binary_crossentropy",
    metrics=["accuracy"],  # , learning_rate=1e-2, optimizer="adam",
)
# Fit to training data
model.fit(
    X_train,
    y_train,
    epochs=1000,
    batch_size=200,
    workers=2,
    shuffle=True,
    use_multiprocessing=True,
    validation_data=(X_test, y_test),
)
