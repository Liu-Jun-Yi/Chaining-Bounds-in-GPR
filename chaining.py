
import scipy.io
import torch
import numpy as np


import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
nreps = 100
delta = 0.1
mu = 1-delta


from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, Matern
import time
import psutil

# Kernel induced distance calculation (matrix version)
def kernel_induced_distance(X, kernel):
    K = kernel(X, X)  
    diag = np.diag(K)[:, None] 
    dist_matrix = np.sqrt(diag + diag.T - 2 * K) 
    return dist_matrix

# Kernel induced distance calculation (point version)
def kernel_induced_distance_single(X, x_new, kernel):
    x_new = np.atleast_2d(x_new)  # 
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
