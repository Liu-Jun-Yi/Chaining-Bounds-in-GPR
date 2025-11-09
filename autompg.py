
import scipy.io
import torch
import numpy as np
import pandas as pd
np.random.seed(2)

nreps = 100
delta = 0.1
datasizes = [800, 5000, 10000]


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", 
           "acceleration", "model_year", "origin", "car_name"]


auto_mpg = pd.read_csv(url, delim_whitespace=True, names=columns, na_values="?")


auto_mpg = auto_mpg.dropna().drop(columns=["car_name"])


X = auto_mpg.drop(columns=["mpg"]).values
y = auto_mpg["mpg"].values


def get_data(ndata):


    n = len(X)

    perm = np.random.permutation(n)

    train_size = int(n * 0.7)

    X_train = X[perm[:train_size]]
    y_train = y[perm[:train_size]]
    X_test = X[perm[train_size:]]
    y_test = y[perm[train_size:]]


    scaler_X = StandardScaler().fit(X_train)
    xs_train = scaler_X.transform(X_train)
    xs_test = scaler_X.transform(X_test)

    scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
    ys_train = scaler_y.transform(y_train.reshape(-1, 1)).flatten()
    ys_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    X_data = xs_train
    y_data = ys_train

    # Load test set
    X_test = xs_test
    f_true = ys_test

    return X_data, y_data, X_test, f_true


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler

import numpy as np
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import time
import psutil
from local_bound import local_bound_RBF,local_bound_Matern
from global_bound import global_bound_RBF,global_bound_Matern
from chaining import construct_partitions, find_partition_diameter_with_mean
from cwc import calculate_picp, calculate_mpiw, calculate_nmpiw, calculate_cwc




url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", 
        "acceleration", "model_year", "origin", "car_name"]


auto_mpg = pd.read_csv(url, delim_whitespace=True, names=columns, na_values="?")

auto_mpg = auto_mpg.dropna().drop(columns=["car_name"])


X = auto_mpg.drop(columns=["mpg"]).values
y = auto_mpg["mpg"].values

n = len(X)


perm = np.random.permutation(n)

train_size = int(n * 0.7)


X_train = X[perm[:train_size]]
y_train = y[perm[:train_size]]
X_test = X[perm[train_size:]]
y_test = y[perm[train_size:]]

scaler_X = StandardScaler().fit(X_train)
X_train = scaler_X.transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
y_train = scaler_y.transform(y_train.reshape(-1, 1)).flatten()
y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

xs_train = X_train
ys_train = y_train 

xs_test = X_test
ys_test = y_test 

picp, mpiw, nmpiw, cwc = local_bound_RBF(xs_train, ys_train, xs_test, ys_test)
print(cwc)
picp, mpiw, nmpiw, cwc = local_bound_Matern(xs_train, ys_train, xs_test, ys_test)
print(cwc)
printpicp, mpiw, nmpiw, cwc = global_bound_RBF(xs_train, ys_train, xs_test, ys_test)
print(cwc)
picp, mpiw, nmpiw, cwc = global_bound_Matern(xs_train, ys_train, xs_test, ys_test)
print(cwc)


