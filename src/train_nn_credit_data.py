import numpy as np
import scipy.stats
import pandas as pd
import sklearn.model_selection as sklms
from main import MultilayerPerceptronClassifier


training_set = np.load("data/credit_data_train.npz")
test_set = np.load("data/credit_data_test.npz")

X_train, y_train = training_set["X_train"], training_set["y_train"].reshape(-1, 1)
X_test, y_test = test_set["X_test"], test_set["y_test"].reshape(-1, 1)

candidate_learning_rates = scipy.stats.uniform(1e-4, 1e-1)
candiate_lambdas = scipy.stats.uniform(0, 1)
param_dist = {"learning_rate": candidate_learning_rates, "lambd": candiate_lambdas}

reg = MultilayerPerceptronClassifier(
    n_epochs=300, batch_size="auto", hidden_layer_size=[100, 50], rtol=1e-2
)

random_search = sklms.RandomizedSearchCV(
    reg,
    n_iter=100,
    param_distributions=param_dist,
    cv=5,
    iid=False,
    n_jobs=-1,
    verbose=True,
    return_train_score=True,
    error_score=np.nan,
)
random_search.fit(X_train, y_train)
random_search.best_estimator_.save_model("nn_credit_model.npz")
df_results = pd.DataFrame.from_dict(random_search.cv_results_, orient="index")
df_results.to_csv("cv_results/results_nn_credit.csv")
