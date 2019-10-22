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
# Marriage
y = y[np.logical_and(X[:, 3] >= 1, X[:, 3] <= 3)]
X = X[np.logical_and(X[:, 3] >= 1, X[:, 3] <= 3)]

# Education
y = y[np.logical_and(X[:, 2] >= 1, X[:, 2] <= 4)]
X = X[np.logical_and(X[:, 2] >= 1, X[:, 2] <= 4)]

# Sex
y = y[np.logical_and(X[:, 1] >= 1, X[:, 1] <= 2)]
X = X[np.logical_and(X[:, 1] >= 1, X[:, 1] <= 2)]


X_trim = pd.DataFrame(X, columns=features.columns.values)

col_encode = list(X_trim.columns.values[1:4])
col_scale = [X_trim.columns.values[0]] + list(X_trim.columns.values[4:])

onehot = sklpre.OneHotEncoder(categories="auto", sparse=False)
scaler = sklpre.StandardScaler()

encoded_cols = onehot.fit_transform(X_trim[col_encode])
scaled_cols = scaler.fit_transform(X_trim[col_scale])

X_trimmed = np.concatenate([encoded_cols, scaled_cols], axis=1)

# Add on intercept column
X_final = np.append(np.ones_like(X_trimmed[:, 1]).reshape(-1, 1), X_trimmed, axis=1)
y_final = y


np.save("data/design_matrix_credit.npy", X_final)
np.save("data/targets_credit.npy", y_final)
