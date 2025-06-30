# Early Stopping for Regression Trees

This repository contains the implementation code for the paper [**"Early Stopping for Regression Trees"**](https://arxiv.org/abs/2502.04709) by Ratmir Miftachov and Markus Reiß.

## Abstract

We develop early stopping rules for growing regression tree estimators. The fully data-driven stopping rule is based on monitoring the global residual norm. The best-first search and the breadth-first search algorithms together with linear interpolation give rise to generalized projection or regularization flows. A general theory of early stopping is established. Oracle inequalities for the early-stopped regression tree are derived without any smoothness assumption on the regression function, assuming the original CART splitting rule, yet with a much broader scope. The remainder terms are of smaller order than the best achievable rates for Lipschitz functions in dimension d≥2. In real and synthetic data the early stopping regression tree estimators attain the statistical performance of cost-complexity pruning while significantly reducing computational costs.

## Repository Structure

```
early-stopping-regression-tree/
├── README.md
├── requirements.txt               # Python dependencies
├── src/
│   ├── regression_tree.py         # Main regression tree implementation with early stopping
│   └── data_generation.py         # Data generating processes for simulations
├── data/
│   ├── Boston.csv                 # Boston housing dataset
│   ├── communities.csv            # Communities and crime dataset
│   ├── ozone_clean.csv           # Ozone dataset
│   └── Servo.csv                 # Servo dataset
├── experiments/
│   ├── empirical_application.py   # Empirical evaluation on real datasets
│   ├── simulation_MSE_tables.py   # MSE comparison tables
│   └── simulation_rel_eff_boxplots.py  # Relative efficiency boxplots
├── visualization/
│   ├── low_dim_heatmaps.py       # Visualization of low-dimensional examples
│   └── tree_stopping_gif.py      # Animation of tree growth with early 

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ratmir-miftachov/Early-Stopping-Regression-Trees.git
cd Early-Stopping-Regression-Trees
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

```python
import numpy as np
import pandas as pd
import sys
sys.path.append('src')
import regression_tree as rt

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 5)
y = X[:, 0] + 0.5 * X[:, 1]**2 + 0.1 * np.random.randn(100)

# Create and fit regression tree with early stopping
tree = rt.RegressionTree(design=X, response=y, min_samples_split=2)
tree.iterate(max_depth=10)

# Find optimal stopping point using discrepancy principle
kappa = 0.1  # Critical value (usually estimated with nearest neighbour estimator)
stopping_depth = tree.get_discrepancy_stop(critical_value=kappa)

# Make predictions
X_test = np.random.randn(20, 5)
predictions = tree.predict(X_test, depth=stopping_depth)

print(f"Optimal stopping depth: {stopping_depth}")
print(f"Test predictions: {predictions[:5]}")
```

## Contact

- **Ratmir Miftachov**: contact[at]miftachov.com
- **Markus Reiß**: reismark[at]hu-berlin.de


