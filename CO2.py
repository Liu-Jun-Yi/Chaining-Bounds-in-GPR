
import numpy as np
import torch
from sklearn.datasets import fetch_openml
np.random.seed(2)
nreps = 100
delta = 0.1

lb = torch.as_tensor([1e-10, 1e-10, 1e-5])
ub = torch.as_tensor([1e5, 5*1e12, 1e2])

def load_mauna_loa_atmospheric_co2():
    ml_data = fetch_openml(data_id=41187, as_frame=False)
    months = []
    ppmv_sums = []
    counts = []

    y = ml_data.data[:, 0]
    m = ml_data.data[:, 1]
    month_float = y + (m - 1) / 12
    ppmvs = ml_data.target

    for month, ppmv in zip(month_float, ppmvs):
        if not months or month != months[-1]:
            months.append(month)
            ppmv_sums.append(ppmv)
            counts.append(1)
        else:
            # aggregate monthly sum to produce average
            ppmv_sums[-1] += ppmv
            counts[-1] += 1

    months = np.asarray(months).reshape(-1, 1)
    avg_ppmvs = np.asarray(ppmv_sums) / counts
    return months, avg_ppmvs

def get_data(ndata):
    perm = torch.as_tensor(list(np.random.permutation(list(range(500)))))
    # perm = torch.as_tensor(list(list(range(500))))
    X_tot, y_tot = load_mauna_loa_atmospheric_co2()
    y_mean = y_tot.mean()
    y_tot = y_tot - y_mean

    X_data = X_tot[perm[0:ndata]]
    y_data = y_tot[perm[0:ndata]]

    # dy = 0.5*ub[-1] * np.random.random(y_data.shape)
    # noise = np.random.normal(0, dy)
    # y_data += noise

    X_test = X_tot[perm[ndata + 1:-1]]
    f_true = y_tot[perm[ndata + 1:-1]]

    return X_data, y_data, X_test, f_true


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from local_bound import local_bound_RBF,local_bound_Matern
from global_bound import global_bound_RBF,global_bound_Matern
from chaining import construct_partitions, find_partition_diameter_with_mean
from cwc import calculate_picp, calculate_mpiw, calculate_nmpiw, calculate_cwc

import numpy as np
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import time
import psutil



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


