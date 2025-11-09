
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


from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import time
import psutil
from chaining import construct_partitions, find_partition_diameter_with_mean
from cwc import calculate_picp, calculate_mpiw, calculate_nmpiw, calculate_cwc

from scipy.integrate import quad


def kernel_induced_distance(X, kernel):
    K = kernel(X, X)  # 计算核矩阵
    diag = np.diag(K)[:, None]  # 提取对角线 (k(x, x))
    dist_matrix = np.sqrt(diag + diag.T - 2 * K)  # 核诱导距离公式
    return dist_matrix

# 核诱导距离计算（单点版本）
def kernel_induced_distance_single(X, x_new, kernel):
    x_new = np.atleast_2d(x_new)  # 确保新点是二维数组
    k_xx = kernel(x_new, x_new)[0, 0]  # k(x_new, x_new)
    k_Xx = kernel(X, x_new).ravel()   # k(X, x_new)
    k_XX = np.diag(kernel(X, X))     # k(X, X)
    distances = np.sqrt(k_xx + k_XX - 2 * k_Xx)
    return distances

T_sets = [[0]]
# 分区序列构造（基于最大化最小距离）
# 改进分区构造方法
def construct_partitions(X, kernel, max_levels=3, diameter_threshold=0.5):
    n_samples = X.shape[0]
    partitions = []  # 存储分区
    diameters = []   # 存储直径
    representative_indices = []  # 存储代表点索引

    # 初始分区
    partitions.append([np.arange(n_samples)])  # 整体作为一个分区
    diameters.append([np.max(kernel_induced_distance(X, kernel))])  # 初始分区的直径
    representative_indices.append([0])  # 初始代表点为第一个点

    dist_matrix = kernel_induced_distance(X, kernel)  # 预计算距离矩阵

    for level in range(1, max_levels + 1):
        num_representatives = min(2 ** (2 ** level), n_samples)  # 每层固定代表点数量
        rep_indices = [np.random.choice(n_samples)]  # 初始化代表点集合，随机选一个起始点

        # 改进：使用最大化最小距离选择代表点
        while len(rep_indices) < num_representatives:
            # 计算点到代表点集合的最小距离
            min_distances = np.min(dist_matrix[:, rep_indices], axis=1)
            # 选择距离最远的点加入代表点集合
            next_rep = np.argmax(min_distances)
            if next_rep not in rep_indices:  # 避免重复选择
                rep_indices.append(next_rep)

        representative_indices.append(rep_indices)

        # 分配点到代表点
        nearest_rep = np.argmin(dist_matrix[:, rep_indices], axis=1)  # 找到每个点最近的代表点
        new_partition = [[] for _ in range(len(rep_indices))]
        for i, rep_idx in enumerate(nearest_rep):
            new_partition[rep_idx].append(i)

        # 计算分区直径
        partition_diameters = []
        for subset in new_partition:
            if len(subset) > 1:
                subset_distances = dist_matrix[np.ix_(subset, subset)]
                partition_diameters.append(np.max(subset_distances))
            else:
                partition_diameters.append(0)  # 单点直径为 0

        partitions.append([np.array(p) for p in new_partition if len(p) > 0])
        diameters.append(partition_diameters)
                # 打印当前层的 Tn
        Tn = rep_indices
        T_sets.append(Tn)
        # print(f"Level {level}: Tn = {Tn}")

    return partitions, diameters, representative_indices


from scipy.integrate import quad

def summand(u, n):
    log_value = np.log(2) + (2 ** (n + 1)) * np.log(2) - (u ** 2) * (2 ** (n - 1))
    # log_value = np.log(2) +  np.log(2) - (u ** 2) * (2 ** (n - 1))
    return np.exp(log_value)


def summation(u, num_terms):
    total = 0
    for n in range(1, num_terms):
        total += summand(u, n)
    return total

def density(u, threshold=1, num_terms=10):
    sum_value = summation(u, num_terms)
    if sum_value <= threshold:
        return sum_value
    else:
        return 1

def squared_density(u, threshold=1, num_terms=10):
    sum_value = summation(u, num_terms)
    if sum_value <= threshold:
        return u**2 * sum_value
    else:
        return 1

def global_bound_RBF(xs_train, ys_train, xs_test, ys_test):



    kernel =  RBF(1.0, (1e-4, 1e4))
    # kernel =  C(1.0, (1e-4, 1e4)) * RBF(1.0, (1e-4, 1e4))
    # kernel = C(1.0, (1e-4, 1e4)) * Matern(length_scale=1.0, length_scale_bounds=(1e-4, 1e4), nu=1.5)

 
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gpr.fit(xs_train, ys_train)

    print(gpr.kernel_)


    kernel = gpr.kernel_
    kernel_callable = lambda x, y: kernel(x, y)

    partitions, diameters, representative_indices = construct_partitions(xs_train, kernel_callable, max_levels=3)


    y_pred_test, sigma_test = gpr.predict(xs_test, return_std=True)
    y_pred, sigma = gpr.predict(xs_train, return_std=True)

    sup_distance = np.zeros(len(xs_train))
    # min_idx = np.zeros(len(xs_test))
    # pai_nt = np.zeros(len(xs_test))
    Xpai_nt = np.zeros(len(xs_train))

    mean = np.zeros(len(xs_train))

    level = len(partitions) - 1


 
    max_n = int(np.log2(np.log2(len(xs_train))))  

    T_sets1 = []
    T_sets1.append([])
    for n in range(1, max_n):
        T_sets1.append(T_sets[n-1])

    sum_distances = np.zeros(len(xs_train))

    for i in range(len(xs_train)):
        for n, (tn, tn1) in enumerate(zip(T_sets, T_sets1)):
            if n >= 1:
                dist1=[]
                for s in tn:
                    cov_st = gpr.kernel_(np.array(xs_train[i]).reshape(1,-1),np.array(xs_train[s]).reshape(1,-1))
                    var_s = gpr.kernel_(np.array(xs_train[s]).reshape(1, -1), np.array(xs_train[s]).reshape(1, -1))
                    var_t = gpr.kernel_(np.array(xs_train[i]).reshape(1, -1), np.array(xs_train[i]).reshape(1, -1))
                    dist1.append(np.sqrt(var_s + var_t - 2 * cov_st  )) 
                s1 =  np.argmin(dist1)   
                dist2=[]
                for s in tn1:
                    cov_st = gpr.kernel_(np.array(xs_train[i]).reshape(1,-1),np.array(xs_train[s]).reshape(1,-1))
                    var_s = gpr.kernel_(np.array(xs_train[s]).reshape(1, -1), np.array(xs_train[s]).reshape(1, -1))
                    var_t = gpr.kernel_(np.array(xs_train[i]).reshape(1, -1), np.array(xs_train[i]).reshape(1, -1))
                    dist2.append(np.sqrt(var_s + var_t - 2 * cov_st )) 
                s2 =  np.argmin(dist2)  

                dist=[]
                cov_st = gpr.kernel_(np.array(xs_train[s1]).reshape(1,-1),np.array(xs_train[s2]).reshape(1,-1))
                var_s = gpr.kernel_(np.array(xs_train[s1]).reshape(1, -1), np.array(xs_train[s1]).reshape(1, -1))
                var_t = gpr.kernel_(np.array(xs_train[s2]).reshape(1, -1), np.array(xs_train[s2]).reshape(1, -1))
                dist= np.sqrt(var_s + var_t - 2 * cov_st)  
                sum_distances[i] += 2**(n/2) * dist


    # sup_distance = np.max(sum_distances)
    sup_distance = sum_distances
    # print(f"Supremum of the sum of distances in training set: {sup_distance}")
    # print((1+ np.sqrt(2)) * L * math.sqrt(math.pi/2) *sup_distance[14])





    u_max = 10  
    result_expectation, _ = quad(density, 0, u_max, epsabs=1e-10, epsrel=1e-10)
    result_squared, _ = quad(squared_density, 0, u_max, epsabs=1e-10, epsrel=1e-10)

    # 计算期望、平方期望和方差
    E_sup_xt = result_expectation * sup_distance  # 数组
    E_sup_xt_squared = result_squared * sup_distance  # 数组

    # 对每个点计算方差
    variance = np.maximum(E_sup_xt_squared - E_sup_xt**2, 0)

    # 如果需要单个全局方差，可以对方差数组取均值
    global_variance = np.mean(variance)

    z = 2.576  
    upper_bound = E_sup_xt + z * np.sqrt(variance)



    error=ys_train[T_sets[0]][0]
    # print(ys_train[T_sets[0]][0])
    error=0
    # print(upper_bound , error)
    upper_bound = upper_bound + error
    lower_bound = - upper_bound + error

    upper_bound = np.max(upper_bound) 
    lower_bound = np.min(lower_bound) 

    ys_test = ys_test.flatten()

    picp = calculate_picp(ys_test, lower_bound, upper_bound)
    mpiw = calculate_mpiw(lower_bound, upper_bound)
    nmpiw = calculate_nmpiw(lower_bound, upper_bound, ys_test)
    eta = 50 
    cwc = calculate_cwc(ys_test, lower_bound, upper_bound, mu, eta)

    return picp, mpiw, nmpiw, cwc


def matern_kernel_3_2(x1, x2, length_scale, constant_value):
    # r = np.linalg.norm(x1 - x2)
    r = abs(x1 - x2)
    sqrt_3_r_by_l = np.sqrt(3) * r / length_scale
    return constant_value * (1 + sqrt_3_r_by_l) * (np.exp(-sqrt_3_r_by_l) - 0.5 )


def global_bound_Matern(xs_train, ys_train, xs_test, ys_test):


    def kernel_induced_distance(X, kernel):
        K = kernel(X, X)  # 计算核矩阵
        diag = np.diag(K)[:, None]  # 提取对角线 (k(x, x))
        dist_matrix = np.sqrt(diag + diag.T - 2 * K)  # 核诱导距离公式
        return dist_matrix

    # 核诱导距离计算（单点版本）
    def kernel_induced_distance_single(X, x_new, kernel):
        x_new = np.atleast_2d(x_new)  # 确保新点是二维数组
        k_xx = kernel(x_new, x_new)[0, 0]  # k(x_new, x_new)
        k_Xx = kernel(X, x_new).ravel()   # k(X, x_new)
        k_XX = np.diag(kernel(X, X))     # k(X, X)
        distances = np.sqrt(k_xx + k_XX - 2 * k_Xx)
        return distances

    T_sets = [[0]]
    # 分区序列构造（基于最大化最小距离）
    # 改进分区构造方法
    def construct_partitions(X, kernel, max_levels=3, diameter_threshold=0.5):
        n_samples = X.shape[0]
        partitions = []  # 存储分区
        diameters = []   # 存储直径
        representative_indices = []  # 存储代表点索引

        # 初始分区
        partitions.append([np.arange(n_samples)])  # 整体作为一个分区
        diameters.append([np.max(kernel_induced_distance(X, kernel))])  # 初始分区的直径
        representative_indices.append([0])  # 初始代表点为第一个点

        dist_matrix = kernel_induced_distance(X, kernel)  # 预计算距离矩阵

        for level in range(1, max_levels + 1):
            num_representatives = min(2 ** (2 ** level), n_samples)  # 每层固定代表点数量
            rep_indices = [np.random.choice(n_samples)]  # 初始化代表点集合，随机选一个起始点

            # 改进：使用最大化最小距离选择代表点
            while len(rep_indices) < num_representatives:
                # 计算点到代表点集合的最小距离
                min_distances = np.min(dist_matrix[:, rep_indices], axis=1)
                # 选择距离最远的点加入代表点集合
                next_rep = np.argmax(min_distances)
                if next_rep not in rep_indices:  # 避免重复选择
                    rep_indices.append(next_rep)

            representative_indices.append(rep_indices)

            # 分配点到代表点
            nearest_rep = np.argmin(dist_matrix[:, rep_indices], axis=1)  # 找到每个点最近的代表点
            new_partition = [[] for _ in range(len(rep_indices))]
            for i, rep_idx in enumerate(nearest_rep):
                new_partition[rep_idx].append(i)

            # 计算分区直径
            partition_diameters = []
            for subset in new_partition:
                if len(subset) > 1:
                    subset_distances = dist_matrix[np.ix_(subset, subset)]
                    partition_diameters.append(np.max(subset_distances))
                else:
                    partition_diameters.append(0)  # 单点直径为 0

            partitions.append([np.array(p) for p in new_partition if len(p) > 0])
            diameters.append(partition_diameters)
                    # 打印当前层的 Tn
            Tn = rep_indices
            T_sets.append(Tn)
            # print(f"Level {level}: Tn = {Tn}")

        return partitions, diameters, representative_indices

    
    T_sets = [[0]]
    kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-4, 1e4), nu=1.5)

    # 拟合高斯过程回归模型
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gpr.fit(xs_train, ys_train)

    print(gpr.kernel_)


    kernel = gpr.kernel_
    kernel_callable = lambda x, y: kernel(x, y)

    # 修正后的分区构造
    partitions, diameters, representative_indices = construct_partitions(xs_train, kernel_callable, max_levels=3)

    sup_distance = np.zeros(len(xs_train))
    # min_idx = np.zeros(len(xs_test))
    # pai_nt = np.zeros(len(xs_test))
    Xpai_nt = np.zeros(len(xs_train))

    mean = np.zeros(len(xs_train))

    level = len(partitions) - 1


    # 计算期望值
    max_n = int(np.log2(np.log2(len(xs_train))))  # 根据数据集大小确定n的上限

    T_sets1 = []
    T_sets1.append([])
    for n in range(1, max_n):
        T_sets1.append(T_sets[n-1])



    fitted_length_scale  = gpr.kernel_.length_scale
    constant_value = 1

    sum_distances = np.zeros(len(xs_train))

    for i in range(len(xs_train)):
        for n, (tn, tn1) in enumerate(zip(T_sets, T_sets1)):
            if n >= 1:
                dist1=[]
                for s in tn:
                    cov_st = matern_kernel_3_2(np.array(xs_train[i]).reshape(1,-1),np.array(xs_train[s]).reshape(1,-1),fitted_length_scale, constant_value)
                    var_s = matern_kernel_3_2(np.array(xs_train[s]).reshape(1,-1), np.array(xs_train[s]).reshape(1,-1),fitted_length_scale, constant_value)
                    var_t = matern_kernel_3_2(np.array(xs_train[i]).reshape(1,-1), np.array(xs_train[i]).reshape(1,-1),fitted_length_scale, constant_value)
                    dist1.append(  np.sqrt(var_s + var_t - 2 * cov_st  ))
                s1 =  np.argmin(dist1)   
                dist2=[]
                for s in tn1:
                    cov_st = matern_kernel_3_2(np.array(xs_train[i]).reshape(1,-1),np.array(xs_train[s]).reshape(1,-1),fitted_length_scale, constant_value)
                    var_s = matern_kernel_3_2(np.array(xs_train[s]).reshape(1,-1), np.array(xs_train[s]).reshape(1,-1),fitted_length_scale, constant_value)
                    var_t = matern_kernel_3_2(np.array(xs_train[i]).reshape(1,-1), np.array(xs_train[i]).reshape(1,-1),fitted_length_scale, constant_value)
                    dist2.append(  np.sqrt(var_s + var_t - 2 * cov_st  ))
                s2 =  np.argmin(dist2)  

                dist=[]
                cov_st = gpr.kernel_(np.array(xs_train[s1]).reshape(1,-1),np.array(xs_train[s2]).reshape(1,-1))
                var_s = gpr.kernel_(np.array(xs_train[s1]).reshape(1, -1), np.array(xs_train[s1]).reshape(1, -1))
                var_t = gpr.kernel_(np.array(xs_train[s2]).reshape(1, -1), np.array(xs_train[s2]).reshape(1, -1))
                dist= np.sqrt(var_s + var_t - 2 * cov_st )
                sum_distances[i] += 2**(n/2) * dist

    # sup_distance = np.max(sum_distances)
    sup_distance = sum_distances
    # print(f"Supremum of the sum of distances in training set: {sup_distance}")
    # print((1+ np.sqrt(2)) * L * math.sqrt(math.pi/2) *sup_distance[14])


    # import numpy as np
    # import scipy.integrate as integrate


    # import numpy as np
    # from scipy.integrate import quad


    # def squared_summand(u, n):
    #     return 2 * u * (2 ** (2 ** (n + 1))) * np.exp((-u ** 2) * (2 ** (n - 1)))

    # def squared_summation(u, num_terms):
    #     total = 0
    #     for n in range(1, num_terms):
    #         total += squared_summand(u, n)
    #     return total

    # def squared_density(u, threshold=1, num_terms=10):
    #     sum_value = squared_summation(u, num_terms)
    #     if sum_value <= threshold:
    #         return sum_value
    #     else:
    #         return 1  # 超过阈值的区间保持为 1

    # # upper_bound =   result * sup_distance 
    # # print(np.max(upper_bound))

    # import numpy as np
    # from scipy.integrate import quad


    # def summand(u, n):
    #     log_value = np.log(2) + (2 ** (n + 1)) * np.log(2) - (u ** 2) * (2 ** (n - 1))
    #     return np.exp(log_value)

    # def summation(u, num_terms):
    #     total = 0
    #     for n in range(1, num_terms):
    #         total += summand(u, n)
    #     return total

    # def density(u, threshold=1, num_terms=10):
    #     sum_value = summation(u, num_terms)
    #     if sum_value <= threshold:
    #         return sum_value
    #     else:
    #         return 1

    # def squared_density(u, threshold=1, num_terms=10):
    #     sum_value = summation(u, num_terms)
    #     if sum_value <= threshold:
    #         return u**2 * sum_value
    #     else:
    #         return 1

    u_max = 100 
    result_expectation, _ = quad(density, 0, u_max, epsabs=1e-10, epsrel=1e-10)
    result_squared, _ = quad(squared_density, 0, u_max, epsabs=1e-10, epsrel=1e-10)

    E_sup_xt = result_expectation * sup_distance  
    E_sup_xt_squared = result_squared * sup_distance  

    variance = np.maximum(E_sup_xt_squared - E_sup_xt**2, 0)

   
    global_variance = np.mean(variance)

    z = 2.576  # 99% 
    # z = 1.96 # 95%
    # z = 1.645  # 90%
    upper_bound = E_sup_xt + z * np.sqrt(variance)



    import matplotlib.pyplot as plt

    error=ys_train[T_sets[0]][0]
    error=0
    upper_bound = upper_bound + error
    lower_bound = - upper_bound + error

    upper_bound = np.max(upper_bound) 
    lower_bound = np.min(lower_bound) 


    ys_test = ys_test.flatten()


    picp = calculate_picp(ys_test, lower_bound, upper_bound)
    mpiw = calculate_mpiw(lower_bound, upper_bound)
    nmpiw = calculate_nmpiw(lower_bound, upper_bound, ys_test)
    # mu = 0.999  # 例如，置信水平为90%
    eta = 50  # 一个较大的值，用于放大PICP和mu之间的差异
    cwc = calculate_cwc(ys_test, lower_bound, upper_bound, mu, eta)
    
    return picp, mpiw, nmpiw, cwc
