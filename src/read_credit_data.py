import os 
import numpy as np 
import pandas as pd 

# Retrieve credit card data
cwd = os.getcwd()
filename = cwd + '/data/default of credit card clients.xls'
df = pd.read_excel(filename, header=1)

# Define design matrix X and targets y
features = df.loc[:, df.columns != 'default payment next month']
X = np.zeros_like(features)
X[:,0] = 1
X[:,1:] = features.loc[:, features.columns != 'ID'].values

y = df["default payment next month"].values

# Export X and y as numpy arrays 
np.save("data/design_matrix_credit.npy", X)
np.save("data/targets_credit.npy", y)
