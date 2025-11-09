
import scipy.io
import torch
import numpy as np


import requests
import pandas as pd
import numpy as np
import scipy.io
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from chaining import construct_partitions, find_partition_diameter_with_mean
from cwc import calculate_picp, calculate_mpiw, calculate_nmpiw, calculate_cwc
nreps = 100
delta = 0.1
mu = 1-delta

np.random.seed(2)
kernel = RBF(length_scale=0.2)
noise_level = 0.5
noise_level_train = noise_level
rkhs_norm = 2
n_samples = 50
n_kernels = 200

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import RBF, Matern
import time
import psutil



# Kernel induced distance calculation (matrix version)
def kernel_induced_distance(X, kernel):
    K = kernel(X, X)  
    diag = np.diag(K)[:, None]  
    dist_matrix = np.sqrt(diag + diag.T - 2 * K) 
    return dist_matrix

# Kernel induced distance calculation (point version)）
def kernel_induced_distance_single(X, x_new, kernel):
    x_new = np.atleast_2d(x_new)  
    k_xx = kernel(x_new, x_new)[0, 0]  # k(x_new, x_new)
    k_Xx = kernel(X, x_new).ravel()   # k(X, x_new)
    k_XX = np.diag(kernel(X, X))     # k(X, X)
    distances = np.sqrt(k_xx + k_XX - 2 * k_Xx)
    return distances

# Partition sequence construction
def construct_partitions(X, kernel, max_levels=3):
    n_samples = X.shape[0]
    partitions = []  
    diameters = []   
    representative_indices = []  

    partitions.append([np.arange(n_samples)])  
    diameters.append([np.max(kernel_induced_distance(X, kernel))])  
    representative_indices.append([0])  

    dist_matrix = kernel_induced_distance(X, kernel)  

    for level in range(1, max_levels + 1):
        num_representatives = min(2 ** (2 ** level), n_samples) 
        rep_indices = [np.random.choice(n_samples)]  
        
        while len(rep_indices) < num_representatives:
            min_distances = np.min(dist_matrix[:, rep_indices], axis=1)
            next_rep = np.argmax(min_distances)
            rep_indices.append(next_rep)

        representative_indices.append(rep_indices)

        nearest_rep = np.argmin(dist_matrix[:, rep_indices], axis=1)  
        new_partition = [[] for _ in range(len(rep_indices))]
        for i, rep_idx in enumerate(nearest_rep):
            new_partition[rep_idx].append(i)

        # Calculate the partition diameter
        partition_diameters = []
        for subset in new_partition:
            if len(subset) > 1:
                subset_distances = dist_matrix[np.ix_(subset, subset)]
                partition_diameters.append(np.max(subset_distances))
            else:
                partition_diameters.append(0) 

        partitions.append([np.array(p) for p in new_partition if len(p) > 0])
        diameters.append(partition_diameters)

    return partitions, diameters, representative_indices


def find_partition_diameter_with_mean(X, y, x_new, partitions, representative_indices, kernel, level):

    x_new = np.atleast_2d(x_new) 

    if level >= len(partitions):
        raise ValueError(f"Level {level} exceeds the maximum partition level {len(partitions) - 1}.")

    rep_indices = representative_indices[level]
    X_representatives = X[rep_indices]

    distances = kernel_induced_distance_single(X_representatives, x_new, kernel)

    nearest_rep_idx = np.argmin(distances)
    nearest_rep = rep_indices[nearest_rep_idx]

    for i, partition in enumerate(partitions[level]):
        if nearest_rep in partition:
            original_y_values = y[partition]
            original_mean_y = np.mean(original_y_values)

            partition_with_new_point = np.append(partition, len(X))  

            X_with_new_point = np.vstack([X, x_new])

            subset = X_with_new_point[partition_with_new_point]
            subset_distances = kernel_induced_distance(subset, kernel)
            diameter = np.max(subset_distances)

            return partition_with_new_point, diameter, original_mean_y

    raise ValueError("Could not find the partition for the new point.")


def sample_rkhs_func_from_kernels(xs, rkhs_norm, n_max_kernels, kernel, n_min_kernels=5):
    n_kernels = np.random.randint(low=n_min_kernels, high=n_max_kernels, size=1)
    indices = np.random.choice(np.arange(xs.size), size=n_kernels, replace=False)
    coeffs = np.random.normal(size=n_kernels)
    K = kernel(xs[indices].reshape([-1,1]), xs[indices].reshape([-1,1]))
    quad_form_val = coeffs.reshape([1,-1]) @ K @ coeffs
    coeffs /= np.sqrt(quad_form_val)
    ys = kernel(xs.reshape([-1,1]), xs[indices].reshape([-1,1])) @ coeffs.reshape([-1,1])
    return ys



def dataset_generation_uniform_normal(xs, ys, n_samples, noise_level_data):
    n_eval = len(xs)
    train_indices = np.random.choice(np.arange(n_eval), n_samples, replace=False)
    xs_train = xs[train_indices]
    ys_train = ys[train_indices] + np.random.normal(loc=0, scale=noise_level_data, size=n_samples)[:,None]
    
    return (xs_train, ys_train)



xs = np.linspace(-1, 1, 1000)
# Build function
ys = sample_rkhs_func_from_kernels(xs, rkhs_norm, n_kernels, kernel)


# Build one training set
(xs_train, ys_train) = dataset_generation_uniform_normal(xs[:500], ys[:500], n_samples, noise_level)
(xs_test, ys_test) = dataset_generation_uniform_normal(xs[500:], ys[500:], n_samples, noise_level)

# xs_train = np.sort(xs_train, axis=0) 
sorted_indices = np.argsort(xs_test.flatten())  
xs_test = xs_test[sorted_indices]  
ys_test = ys_test[sorted_indices]

kernel =  C(1.0, (1e-4, 1e4)) * RBF(1.0, (1e-4, 1e4))
# kernel = C(1.0, (1e-4, 1e4)) * Matern(length_scale=1.0, length_scale_bounds=(1e-4, 1e4), nu=1.5)

gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gpr.fit(xs_train.reshape(-1,1), ys_train)

print(gpr.kernel_)


kernel = gpr.kernel_
kernel_callable = lambda x, y: kernel(x, y)

partitions, diameters, representative_indices = construct_partitions(xs_train.reshape(-1,1), kernel_callable, max_levels=3)


sup_distance = np.zeros(len(xs_test))
min_idx = np.zeros(len(xs_test))
pai_nt = np.zeros(len(xs_test))
Xpai_nt = np.zeros(len(xs_test))

mean = np.zeros(len(xs_test))

level = len(partitions) - 1

for i in range(len(xs_test)):
    partition, diameter, original_mean_y = find_partition_diameter_with_mean(xs_train.reshape(-1,1), ys_train, xs_test[i], partitions, representative_indices, kernel_callable, level)
    sup_distance[i] = diameter
    Xpai_nt[i] = original_mean_y



mu = 1-delta
u = sup_distance * np.sqrt(2 * (np.log(1/delta) + np.log(2)))

upper_bound =   Xpai_nt + u
lower_bound =   Xpai_nt - u


ys_test = ys_test.flatten()


picp, uncovered_indices = calculate_picp(ys_test, lower_bound, upper_bound)
mpiw = calculate_mpiw(lower_bound, upper_bound)
nmpiw = calculate_nmpiw(lower_bound, upper_bound, ys_test)
eta = 50  
cwc = calculate_cwc(ys_test, lower_bound, upper_bound, mu, eta)



kernel = C(1.0, (1e-4, 1e4)) * Matern(length_scale=1.0, length_scale_bounds=(1e-4, 1e4), nu=0.5)


gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gpr.fit(xs_train.reshape(-1,1), ys_train)

print(gpr.kernel_)


kernel = gpr.kernel_
kernel_callable = lambda x, y: kernel(x, y)


partitions, diameters, representative_indices = construct_partitions(xs_train.reshape(-1,1), kernel_callable, max_levels=3)


sup_distance = np.zeros(len(xs_test))
min_idx = np.zeros(len(xs_test))
pai_nt = np.zeros(len(xs_test))
Xpai_nt = np.zeros(len(xs_test))

mean = np.zeros(len(xs_test))

level = len(partitions) - 1

for i in range(len(xs_test)):
    partition, diameter, original_mean_y = find_partition_diameter_with_mean(xs_train.reshape(-1,1), ys_train, xs_test[i], partitions, representative_indices, kernel_callable, level)
    sup_distance[i] = diameter
    Xpai_nt[i] = original_mean_y



mu = 1-delta
u = sup_distance * np.sqrt(2 * (np.log(1/delta) + np.log(2)))

upper_bound =   Xpai_nt + u
lower_bound =   Xpai_nt - u


picp, uncovered_indices = calculate_picp(ys_test, lower_bound, upper_bound)
mpiw = calculate_mpiw(lower_bound, upper_bound)
nmpiw = calculate_nmpiw(lower_bound, upper_bound, ys_test)
eta = 50  
cwc = calculate_cwc(ys_test, lower_bound, upper_bound, mu, eta)
