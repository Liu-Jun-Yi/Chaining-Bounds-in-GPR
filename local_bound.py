
import scipy.io
import torch
import numpy as np


import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
nreps = 100
delta = 0.1
mu = 1-delta

from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from sklearn.gaussian_process.kernels import RBF, Matern

import time
import psutil
from chaining import construct_partitions, find_partition_diameter_with_mean
from cwc import calculate_picp, calculate_mpiw, calculate_nmpiw, calculate_cwc


def local_bound_RBF(xs_train, ys_train, xs_test, ys_test):

    kernel =  C(1.0, (1e-4, 1e4)) * RBF(1.0, (1e-4, 1e4))


    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gpr.fit(xs_train, ys_train)

    print(gpr.kernel_)


    kernel = gpr.kernel_
    kernel_callable = lambda x, y: kernel(x, y)

    partitions, diameters, representative_indices = construct_partitions(xs_train, kernel_callable, max_levels=3)


    sup_distance = np.zeros(len(xs_test))
    min_idx = np.zeros(len(xs_test))
    pai_nt = np.zeros(len(xs_test))
    Xpai_nt = np.zeros(len(xs_test))

    mean = np.zeros(len(xs_test))

    level = len(partitions) - 1

    for i in range(len(xs_test)):
        partition, diameter, original_mean_y = find_partition_diameter_with_mean(xs_train, ys_train, xs_test[i], partitions, representative_indices, kernel_callable, level)
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

    return picp, mpiw, nmpiw, cwc


def local_bound_Matern(xs_train, ys_train, xs_test, ys_test):
    # kernel = C(1.0, (1e-4, 1e4)) * Matern(length_scale=1.0, length_scale_bounds=(1e-4, 1e4), nu=1.5)
    kernel = C(1.0, (1e-4, 1e4)) * Matern(length_scale=1.0, length_scale_bounds=(1e-4, 1e4), nu=0.5)

    # 拟合高斯过程回归模型
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gpr.fit(xs_train, ys_train)

    print(gpr.kernel_)


    kernel = gpr.kernel_
    kernel_callable = lambda x, y: kernel(x, y)

    partitions, diameters, representative_indices = construct_partitions(xs_train, kernel_callable, max_levels=3)


    sup_distance = np.zeros(len(xs_test))
    min_idx = np.zeros(len(xs_test))
    pai_nt = np.zeros(len(xs_test))
    Xpai_nt = np.zeros(len(xs_test))

    mean = np.zeros(len(xs_test))

    level = len(partitions) - 1

    for i in range(len(xs_test)):
        partition, diameter, original_mean_y = find_partition_diameter_with_mean(xs_train, ys_train, xs_test[i], partitions, representative_indices, kernel_callable, level)
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
    
    return picp, mpiw, nmpiw, cwc