import sys
import numpy as np
import scipy.stats
import pandas as pd
import sklearn.model_selection as sklms
from main import MultilayerPerceptronRegressor

try:
    n_x = int(sys.argv[1])
    n_y = int(sys.argv[2])
    sigma = float(sys.argv[3])
except IndexError:
    raise IndexError(
        f"Please input the number of points in x direction, y direction"
        + f" and the standard deviation of the generated data you wish to model"
    )
except ValueError:
    raise TypeError("Input must be integer, integer and float")


training_set = np.load(f"data/franke_data_train_{n_x}_{n_y}_{sigma}.npz")
test_set = np.load(f"data/franke_data_test_{n_x}_{n_y}_{sigma}.npz")

X_train, z_train = training_set["X_train"], training_set["z_train"]
X_test, z_test = test_set["X_test"], test_set["z_test"]


reg = MultilayerPerceptronRegressor(
    n_epochs=300,
    batch_size="auto",
    hidden_layer_size=[100, 50],
    rtol=1e-2,
    verbose=False,
    activation_function_output="linear",
)

candidate_learning_rates = scipy.stats.uniform(1e-4, 1e-2)
candiate_lambdas = scipy.stats.uniform(0, 1)
param_dist = {"learning_rate": candidate_learning_rates, "lambd": candiate_lambdas}

random_search = sklms.RandomizedSearchCV(
    reg,
    n_iter=100,
    scoring="r2",
    param_distributions=param_dist,
    cv=5,
    iid=False,
    n_jobs=-1,
    verbose=True,
    return_train_score=True,
)

random_search.fit(X_train, z_train)
random_search.best_estimator_.save_model(f"franke_model_{n_x}_{n_y}_{sigma}.npz")
df_results = pd.DataFrame.from_dict(random_search.cv_results_, orient="index")
df_results.to_csv(f"cv_results/results_nn_franke_{n_x}_{n_y}_{sigma}.csv")
