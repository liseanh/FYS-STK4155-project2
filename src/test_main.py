import numpy as np 
from main import MultilayerPerceptronClassifier, MultilayerPerceptronRegressor
import sklearn.preprocessing as sklpre
import matplotlib.pyplot as plt

def test_MultilayerPerceptronRegressor():
    x = np.linspace(0, 5, 10000, endpoint=True)
    y = 3*x + 2
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    print(x.shape)
    scalerx = sklpre.StandardScaler().fit(x)
    x = scalerx.transform(x)

    scalery = sklpre.MinMaxScaler().fit(y)
    y = scalery.transform(y)

    model = MultilayerPerceptronRegressor(hidden_layer_size=[1], rtol=-np.inf, learning_rate=1e-3)
    model.fit(x, y)
    y_model = model.predict(x)
    plt.plot(x, y, x, y_model)
    plt.show()
    #print(y_model-y)
    #np.testing.assert_array_almost_equal(y, y_model, decimal=3)

test_MultilayerPerceptronRegressor()