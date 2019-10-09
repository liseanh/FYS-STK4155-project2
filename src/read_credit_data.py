import os
import numpy as np
import pandas as pd
import sklearn.preprocessing as sklpre
import sklearn.compose as sklco

# Retrieve credit card data
cwd = os.getcwd()
filename = cwd + "/data/default of credit card clients.xls"
df = pd.read_excel(filename, header=1)

# Define design matrix X and targets y
features = df.loc[:, (df.columns != "default payment next month") & (df.columns != "ID")]
X = features.values
y = df["default payment next month"].values

# Removing feature outliers
y = y[np.logical_and(X[:, 3] >= 1, X[:, 3] <= 3)]
X = X[np.logical_and(X[:, 3] >= 1, X[:, 3] <= 3)]

y = y[np.logical_and(X[:, 2] >= 1, X[:, 2] <= 4)]
X = X[np.logical_and(X[:, 2] >= 1, X[:, 2] <= 4)]

y = y[np.logical_and(X[:, 1] >= 1, X[:, 1] <= 2)]
X = X[np.logical_and(X[:, 1] >= 1, X[:, 1] <= 2)]

"""
for i in range(5, 11):
    y = y[np.logical_and(np.logical_and(X[:, i] >= -1, X[:, i] <= 9), X[:, i] != 0)]
    X = X[np.logical_and(np.logical_and(X[:, i] >= -1, X[:, i] <= 9), X[:, i] != 0)]
"""

# Onehotting categorical features
onehot = sklpre.OneHotEncoder(categories="auto")
X = sklco.ColumnTransformer(
    [("", onehot, [1, 2, 3])], remainder="passthrough").fit_transform(X)

X = np.append(np.ones_like(X[:, 1]).reshape(-1, 1), X, axis=1)

# Export X and y as numpy arrays
np.save("data/design_matrix_credit.npy", X)
np.save("data/targets_credit.npy", y)
