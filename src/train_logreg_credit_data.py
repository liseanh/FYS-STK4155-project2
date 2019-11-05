import numpy as np
import pandas as pd
from main import LogisticRegression
import sklearn.preprocessing as sklpre
import sklearn.model_selection as sklms
import scipy.stats
np.random.seed(len("nama jeff"))

training_set = np.load("data/credit_data_train.npz")
test_set = np.load("data/credit_data_test.npz")

X_train, y_train = training_set["X_train"], training_set["y_train"].reshape(-1, 1)
X_test, y_test = test_set["X_test"], test_set["y_test"].reshape(-1, 1)

reg = LogisticRegression(n_epochs=1000, rtol=0.01, batch_size="auto")
candidate_learning_rates = scipy.stats.uniform(1e-5, 1e-2)
param_dist = {"learning_rate": candidate_learning_rates}
random_search = sklms.RandomizedSearchCV(
    reg,
    n_iter=100,
    param_distributions=param_dist,
    cv=5,
    iid=False,
    n_jobs=-1,
    verbose=True,
    return_train_score=True,
)
random_search.fit(X_train, y_train)
random_search.best_estimator_.save_model("logreg_credit_model.npz")
df_results = pd.DataFrame.from_dict(random_search.cv_results_, orient="index")
df_results.to_csv("cv_results/results_logreg.csv")
