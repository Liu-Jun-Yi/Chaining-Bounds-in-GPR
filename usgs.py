import scipy.io
import torch
import numpy as np

np.random.seed(2)
import requests
import pandas as pd
import numpy as np
import scipy.io
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from local_bound import local_bound_RBF,local_bound_Matern
from global_bound import global_bound_RBF,global_bound_Matern
from chaining import construct_partitions, find_partition_diameter_with_mean
from cwc import calculate_picp, calculate_mpiw, calculate_nmpiw, calculate_cwc
# system and simulation parameters
dimx = 3  # 例如：经度、纬度和深度作为特征
ndata_max = 10000
ndata_min = 800
ntest = 4000
num_samples = 100
training_iterations = 10000
datasizes = [ndata_min, 5000, ndata_max]  # 数据集大小可以根据需要设置
nreps = 100
delta = 0.01
mu = 1- delta

def get_data(ndata):
    # 从 USGS API 获取地震数据
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": "2020-01-01",
        "endtime": "2024-12-31",
        "minlatitude": 34.0,
        "maxlatitude": 36.0,
        "minlongitude": -120.0,
        "maxlongitude": -118.0,
        "minmagnitude": 3.0,
    }
    response = requests.get(url, params=params)
    data = response.json()

    # 提取地震特征
    earthquake_data = []
    for feature in data['features']:
        properties = feature['properties']
        geometry = feature['geometry']['coordinates']
        earthquake_data.append({
            "longitude": geometry[0],
            "latitude": geometry[1],
            "depth": geometry[2],
            "time": properties["time"],
            "magnitude": properties["mag"],
        })

    df = pd.DataFrame(earthquake_data)

    # 删除含缺失值的行
    df = df.dropna()

    ndata = int(len(df) * 0.7)

    # 随机抽取数据集
    perm = np.random.permutation(len(df))
    X_data = df[['longitude', 'latitude', 'depth']].iloc[perm[:ndata], :].values
    y_data = df['magnitude'].iloc[perm[:ndata]].values

    # 划分测试集
    X_test = df[['longitude', 'latitude', 'depth']].iloc[perm[ndata:ndata+ntest], :].values
    y_true = df['magnitude'].iloc[perm[ndata:ndata+ntest]].values

    return X_data, y_data, X_test, y_true

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
