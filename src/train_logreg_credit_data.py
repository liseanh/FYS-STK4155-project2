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
# reg = LogisticRegression(learning_rate=1e-5, verbose=True, rtol=-np.inf, batch_size=69)
# reg.fit(np.append(np.ones(X_train.shape[0]).reshape(-1, 1), X_train, axis=1), y_train)
# print(reg.accuracy_score(X_test, y_test))

reg = LogisticRegression(n_epochs=1000, rtol=0.01)
candidate_learning_rates = scipy.stats.uniform(1e-5, 1e-2)
candidate_batch_sizes = scipy.stats.randint(1, len(y_test))
param_dist = {"learning_rate": candidate_learning_rates, "batch_size": candidate_batch_sizes}
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
print(random_search.cv_results_)
print(random_search.best_score_)
print(random_search.score(X_test, y_test))
df_results = pd.DataFrame.from_dict(random_search.cv_results_, orient="index")
df_results.to_csv("cv_results/results_logreg.csv")
