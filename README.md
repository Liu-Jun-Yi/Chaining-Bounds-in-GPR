# Practical Global and Local Bounds in Gaussian Process Regression via Chaining
Published as a conference paper at AAAI 2026.

## Abstract
Gaussian process regression (GPR) is a popular nonparametric Bayesian method that provides predictive uncertainty estimates and is widely used in safety-critical applications. While prior research has introduced various uncertainty bounds, most existing approaches 
require access to specific input features and rely on posterior mean and variance estimates or tuning hyperparameters. These limitations hinder robustness and fail to capture the model’s global behavior in expectation.
To address these limitations, we propose a chaining-based framework for estimating upper and lower bounds on the expected extreme values over unseen data, without requiring access to specific input locations. We provide kernel-specific refinements for commonly used kernels such as RBF and Matérn, in which our bounds are tighter than generic constructions. We further improve numerical tightness by avoiding analytical relaxations. 
In addition to global estimation, we also develop a novel method for local uncertainty quantification at specified inputs. This approach leverages chaining geometry through partition diameters, adapting to local structure without relying on posterior variance scaling. 
Our experimental results validate the theoretical findings and demonstrate that our method outperforms existing approaches on both synthetic and real-world datasets.


## Usage

If you want to run the model, run the command as below.

```
python main.py --data dataset
```

## Citing this work
