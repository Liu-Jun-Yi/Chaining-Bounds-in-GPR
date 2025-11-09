
import numpy as np

# 计算PICP
def calculate_picp(y_test, lower_bound, upper_bound):
    coverage = (y_test >= lower_bound) & (y_test <= upper_bound)
    picp = np.mean(coverage)
    uncovered_indices = np.where(~coverage)[0]
    return picp, uncovered_indices

# 计算MPIW
def calculate_mpiw(lower_bound, upper_bound):
    mpiw = np.mean(upper_bound - lower_bound)
    return mpiw

# 计算NMPIW
def calculate_nmpiw(lower_bound, upper_bound, y_test):
    mpiw = calculate_mpiw(lower_bound, upper_bound)
    target_range = np.max(y_test) - np.min(y_test)
    nmpiw = mpiw / target_range
    return nmpiw

# 计算CWC
def calculate_cwc(y_test, lower_bound, upper_bound, mu, eta):
    picp, uncovered_indices = calculate_picp(y_test, lower_bound, upper_bound)
    nmpiw = calculate_nmpiw(lower_bound, upper_bound, y_test)
    
    # 使用阶跃函数定义 γ(PICP)
    gamma = 1 if picp < mu else 0
    
    cwc = nmpiw * (1 + gamma * np.exp(-eta * (picp - mu)))
    return cwc
