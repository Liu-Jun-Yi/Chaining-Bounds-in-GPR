import numpy as np
from sklearn import preprocessing
import urllib.request
import numpy as np
from sklearn.preprocessing import StandardScaler
import urllib.request
import scipy.io
import requests
import pandas as pd

def load_boston():
    """Robust loader for Boston housing data."""
    url = "http://lib.stat.cmu.edu/datasets/boston"
    with urllib.request.urlopen(url) as f:
        lines = f.read().decode().split("\n")[22:]
    data = []
    for i in range(0, len(lines) - 1, 2):
        if lines[i].strip() == "" or lines[i + 1].strip() == "":
            continue
        row1 = list(map(float, lines[i].strip().split()))
        row2 = list(map(float, lines[i + 1].strip().split()))
        data.append(row1 + row2)
    data = np.array(data)
    X, y = data[:, :-1], data[:, -1]
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    return X, y

def load_sarcos(
    torquenr: int = 0,
    train_path: str = "datasets/sarcos_inv.mat",
    test_path: str = "datasets/sarcos_inv_test.mat",
    use_test: bool = False,
):
    train = scipy.io.loadmat(train_path)["sarcos_inv"]
    X_train = train[:, :21]
    y_train = train[:, 22 + torquenr]

    if use_test:
        test = scipy.io.loadmat(test_path)["sarcos_inv_test"]
        X_test = test[:, :21]
        y_test = test[:, 22 + torquenr]
        X = np.vstack([X_train, X_test])
        y = np.concatenate([y_train, y_test])
    else:
        X, y = X_train, y_train

    return X, y

def load_usgs(
    starttime: str = "2020-01-01",
    endtime: str = "2024-12-31",
    minlatitude: float = 34.0,
    maxlatitude: float = 36.0,
    minlongitude: float = -120.0,
    maxlongitude: float = -118.0,
    minmagnitude: float = 3.0,
):
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": starttime,
        "endtime": endtime,
        "minlatitude": minlatitude,
        "maxlatitude": maxlatitude,
        "minlongitude": minlongitude,
        "maxlongitude": maxlongitude,
        "minmagnitude": minmagnitude,
        "orderby": "time-asc",
        "limit": 20000,  # 防止太少；可按需调整
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()

    earthquake_data = []
    for f in data.get("features", []):
        props = f.get("properties", {})
        geom = f.get("geometry", {})
        coords = geom.get("coordinates", None)
        mag = props.get("mag", None)
        if (
            coords is None
            or len(coords) < 3
            or mag is None
        ):
            continue
        earthquake_data.append(
            {
                "longitude": coords[0],
                "latitude": coords[1],
                "depth": coords[2],
                "magnitude": mag,
            }
        )

    if not earthquake_data:
        raise RuntimeError("USGS loader: no valid earthquake data retrieved.")

    df = pd.DataFrame(earthquake_data).dropna()
    if len(df) < 100:
        raise RuntimeError(f"USGS loader: too few samples ({len(df)}) after filtering.")

    X = df[["longitude", "latitude", "depth"]].values
    y = df["magnitude"].values
    return X, y


def load_co2():
    """Mauna Loa atmospheric CO2 dataset (OpenML id=41187)."""
    ml_data = fetch_openml(data_id=41187, as_frame=False)
    years = ml_data.data[:, 0]
    months = ml_data.data[:, 1]
    ppmvs = ml_data.target

    # aggregate monthly
    month_float = years + (months - 1) / 12
    df = pd.DataFrame({"month": month_float, "ppmv": ppmvs})
    df = df.groupby("month", as_index=False).mean()
    X = df[["month"]].values
    y = df["ppmv"].values - np.mean(df["ppmv"].values)
    return X, y

def load_autompg():
    """
    UCI Auto MPG dataset.
    Features: all except mpg, car_name
    Target: mpg
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    columns = [
        "mpg",
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
        "origin",
        "car_name",
    ]
    df = pd.read_csv(
        url,
        delim_whitespace=True,
        names=columns,
        na_values="?",
    )
    df = df.dropna()
    df = df.drop(columns=["car_name"])

    X = df.drop(columns=["mpg"]).values
    y = df["mpg"].values
    return X, y



def load_dataset(name: str):
    """Dataset selector for main.py."""
    name = name.lower()
    if name == "boston":
        return load_boston()
    elif name == "sarcos":
        return load_sarcos()
    elif name == "co2":
        return load_co2()
    elif name == "usgs":
        return load_sarcos()
    elif name == "autompg":
        return load_co2()
    else:
        raise ValueError(f"Unknown dataset: {name}")
