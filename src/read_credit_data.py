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
features = df.loc[:, df.columns != "default payment next month"]
X = np.zeros_like(features)
X[:, 0] = 1
X[:, 1:] = features.loc[:, features.columns != "ID"].values

y = df["default payment next month"].values

# Removing feature outliers
y = y[np.logical_and(X[:, 4] >= 1, X[:, 4] <= 3)]
X = X[np.logical_and(X[:, 4] >= 1, X[:, 4] <= 3)]

y = y[np.logical_and(X[:, 3] >= 1, X[:, 3] <= 4)]
X = X[np.logical_and(X[:, 3] >= 1, X[:, 3] <= 4)]

y = y[np.logical_and(X[:, 2] >= 1, X[:, 2] <= 2)]
X = X[np.logical_and(X[:, 2] >= 1, X[:, 2] <= 2)]

for i in range(6, 12):
    y = y[np.logical_and(np.logical_and(X[:, i] >= -1, X[:, i] <= 9), X[:, i] != 0)]
    X = X[np.logical_and(np.logical_and(X[:, i] >= -1, X[:, i] <= 9), X[:, i] != 0)]

# Onehotting categorical features
onehot = sklpre.OneHotEncoder(categories="auto")
X = sklco.ColumnTransformer(
    [("", onehot, [2, 3, 4])], remainder="passthrough"
).fit_transform(X)

X[:, :10] = np.roll(X[:, :10], 1, axis=1)

# Export X and y as numpy arrays
np.save("data/design_matrix_credit.npy", X)
np.save("data/targets_credit.npy", y)
