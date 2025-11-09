#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler

from dataset_utils import load_dataset
from local_bound import local_bound_RBF, local_bound_Matern
from global_bound import global_bound_RBF, global_bound_Matern


def get_data(X_tot, y_tot, ndata, seed=42):
    rng = np.random.RandomState(seed)
    n = X_tot.shape[0]
    if ndata >= n:
        raise ValueError(f"ndata={ndata} must be < total samples={n}")

    perm = rng.permutation(n)
    X_data = X_tot[perm[:ndata]]
    y_data = y_tot[perm[:ndata]]
    X_test = X_tot[perm[ndata:]]
    f_true = y_tot[perm[ndata:]]
    return X_data, y_data, X_test, f_true


def parse_result(result, name):
    if isinstance(result, dict):
        if all(k in result for k in ("picp", "mpiw", "nmpiw", "cwc")):
            return (
                float(result["picp"]),
                float(result["mpiw"]),
                float(result["nmpiw"]),
                float(result["cwc"]),
            )

    flat = []

    def _flatten(x):
        if isinstance(x, (list, tuple)):
            for v in x:
                _flatten(v)
        elif isinstance(x, dict):
            for v in x.values():
                _flatten(v)
        else:
            flat.append(x)

    _flatten(result)
    nums = [v for v in flat if isinstance(v, (int, float, np.floating))]
    if len(nums) < 4:
        raise ValueError(f"{name} returned unexpected format: {result!r}")
    picp, mpiw, nmpiw, cwc = nums[0], nums[1], nums[2], nums[3]
    return float(picp), float(mpiw), float(nmpiw), float(cwc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="boston",
                        help="Choose dataset: boston / sarcos / usgs / co2 / auto_mpg ...")
    parser.add_argument("--ndata", type=int, default=250,
                        help="Number of training points.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/test split.")
    args = parser.parse_args()

    X_tot, y_tot = load_dataset(args.dataset)
    X_data, y_data, X_test, f_true = get_data(X_tot, y_tot, args.ndata, seed=args.seed)

    scaler_X = StandardScaler().fit(X_data)
    X_train = scaler_X.transform(X_data)
    X_test = scaler_X.transform(X_test)

    scaler_y = StandardScaler().fit(y_data.reshape(-1, 1))
    y_train = scaler_y.transform(y_data.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(f_true.reshape(-1, 1)).flatten()

    for fn, name in [
        (local_bound_RBF, "local_RBF"),
        (local_bound_Matern, "local_Matern"),
        (global_bound_RBF, "global_RBF"),
        (global_bound_Matern, "global_Matern"),
    ]:
        raw = fn(X_train, y_train, X_test, y_test)
        picp, mpiw, nmpiw, cwc = parse_result(raw, name)
        print(f"{name} — CWC: {cwc:.4f}, PICP: {picp:.3f}, MPIW: {mpiw:.3f}, N-MPIW: {nmpiw:.3f}")


if __name__ == "__main__":
    main()
