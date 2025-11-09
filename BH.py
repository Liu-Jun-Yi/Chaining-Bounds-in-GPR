import numpy as np
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler


from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
import time
import psutil
from local_bound import local_bound_RBF,local_bound_Matern
from global_bound import global_bound_RBF,global_bound_Matern
from chaining import construct_partitions, find_partition_diameter_with_mean
from cwc import calculate_picp, calculate_mpiw, calculate_nmpiw, calculate_cwc

from sklearn import preprocessing
import pandas as pd
import urllib.request

# Robust version of manual Boston dataset loader
def load_boston_manual():
    url = "http://lib.stat.cmu.edu/datasets/boston"
    with urllib.request.urlopen(url) as f:
        lines = f.read().decode().split("\n")[22:]  # skip headers

    # Each sample is split across 2 lines
    data = []
    for i in range(0, len(lines) - 1, 2):
        if lines[i].strip() == "" or lines[i + 1].strip() == "":
            continue
        row1 = list(map(float, lines[i].strip().split()))
        row2 = list(map(float, lines[i + 1].strip().split()))
        data.append(row1 + row2)

    data = np.array(data)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

# Load and scale the full dataset
X_tot, y_tot = load_boston_manual()
scaler = preprocessing.StandardScaler().fit(X_tot)
X_tot = scaler.transform(X_tot)

# System and simulation parameters
nmc = 1000
dimx = X_tot.shape[1]
ndata_max = 450
ndata_min = 50
num_samples = 100
warmup_steps = 100
nreps = 100
delta = 0.05
z = 1.96  # 95%

def get_data(ndata):
    perm = list(np.random.permutation(list(range(506))))
    X_data = X_tot[perm[0:ndata]]
    y_data = y_tot[perm[0:ndata]]
    X_test = X_tot[perm[ndata + 1:-1]]
    f_true = y_tot[perm[ndata + 1:-1]]
    return X_data, y_data, X_test, f_true

np.random.seed(42)
X_data, y_data, X_test, f_true = get_data(250)


X_train, y_train, y_test = X_data, y_data, f_true
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



