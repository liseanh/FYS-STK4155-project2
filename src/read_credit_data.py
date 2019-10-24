import os
import numpy as np
import pandas as pd
import sklearn.preprocessing as sklpre
import sklearn.compose as sklco
import sklearn.model_selection as sklms
import imblearn

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


# Oversampling to get equal ratio of targets 0 and 1
ros = imblearn.over_sampling.RandomOverSampler(sampling_strategy=1)
X_resample, y_resample = ros.fit_resample(X, y)


# Split the data into training and test set
X_train_, X_test_, y_train, y_test = sklms.train_test_split(
    X_resample, y_resample, test_size=0.33,
)

# Make dataframes of scaled features
X_train_df = pd.DataFrame(X_train_, columns=features.columns.values)

X_test_df = pd.DataFrame(X_test_, columns=features.columns.values)


# Onehot-encode categotical features
col_encode = list(X_train_df.columns.values[1:4])

onehot = sklpre.OneHotEncoder(categories="auto", sparse=False)

encoded_cols_train = onehot.fit_transform(X_train_df[col_encode])
encoded_cols_test = onehot.fit_transform(X_test_df[col_encode])


# Scale the rest of the features
col_scale = [X_train_df.columns.values[0]] + list(X_train_df.columns.values[4:])

scaler = sklpre.StandardScaler().fit(X_train_df[col_scale])

scaled_cols_train = scaler.transform(X_train_df[col_scale])
scaled_cols_test = scaler.transform(X_test_df[col_scale])

# Combine into one array for training, one for testing
X_train = np.concatenate([encoded_cols_train, scaled_cols_train], axis=1)
X_test = np.concatenate([encoded_cols_test, scaled_cols_test], axis=1)

# Export the preprocessed data
np.savez("data/credit_data_train.npz", X_train=X_train, y_train=y_train)
np.savez("data/credit_data_test.npz", X_train=X_train, y_train=y_train)
