import numpy as np
import torch
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
import scipy
import numpy as np
from sklearn.gaussian_process.kernels import RBF, Matern
import time
import psutil
from local_bound import local_bound_RBF,local_bound_Matern
from global_bound import global_bound_RBF,global_bound_Matern
from chaining import construct_partitions, find_partition_diameter_with_mean
from cwc import calculate_picp, calculate_mpiw, calculate_nmpiw, calculate_cwc

# system and simulation parameters
dimx = 21
nreps = 105
ntest = 4000
ndata_max = 10000
ndata_min = 800
num_samples = 100
warmup_steps = 1
training_iterations = 10000 # 10000
ndatanumbers = 3
torquenr = 0  # specifies which torque is to be inferred (0-6)
nreps =100
datasizes = [800, 5000, 10000]

def get_data(ndata):
    # Load training set
    train = scipy.io.loadmat("/home/jliu/gauss_proc_unknown_hyp-main/datasets/sarcos_inv.mat")
    perm = torch.as_tensor(list(np.random.permutation(list(range(44484)))))
    # Inputs  (7 joint positions, 7 joint velocities, 7 joint accelerations)
    X_data = train["sarcos_inv"][perm[:ndata], :21]
    # Outputs (7 joint torques)
    y_data = train["sarcos_inv"][perm[:ndata], 22 + torquenr]

    # Load test set
    test = scipy.io.loadmat("/home/jliu/gauss_proc_unknown_hyp-main/datasets/sarcos_inv_test.mat")
    X_test = test["sarcos_inv_test"][:ntest, :21]
    f_true = test["sarcos_inv_test"][:ntest, 22 + torquenr]

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
