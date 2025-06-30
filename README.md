# Early Stopping for Regression Trees

This repository contains the implementation code for the paper [**"Early Stopping for Regression Trees"**](https://arxiv.org/abs/2502.04709) by Ratmir Miftachov and Markus Reiß.

## Abstract

We develop early stopping rules for growing regression tree estimators. The fully data-driven stopping rule is based on monitoring the global residual norm. The best-first search and the breadth-first search algorithms together with linear interpolation give rise to generalized projection or regularization flows. A general theory of early stopping is established. Oracle inequalities for the early-stopped regression tree are derived without any smoothness assumption on the regression function, assuming the original CART splitting rule, yet with a much broader scope. The remainder terms are of smaller order than the best achievable rates for Lipschitz functions in dimension d≥2. In real and synthetic data the early stopping regression tree estimators attain the statistical performance of cost-complexity pruning while significantly reducing computational costs.

## Visualization

The animation demonstrates the progression of the regression tree estimator, comparing global and semi-global early stopping approaches on a 2D rectangular function.

![Tree Growth Animation](visualization/tree_stopping_animation.gif)

## Quick Start

```python
import numpy as np
import pandas as pd
import sys
sys.path.append('.')
from src.algorithms import global_regression_tree as rt
from src.utils import noise_level_estimator as noise_est

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 5)
y = X[:, 0] + 0.5 * X[:, 1]**2 + 0.1 * np.random.randn(100)

# Create and fit regression tree with early stopping
tree = rt.RegressionTree(design=X, response=y, min_samples_split=2)
tree.iterate(max_depth=10)

# Estimate noise level (or use known value)
estimator = noise_est.Estimator(X, y)
kappa = estimator.estimate(method='1NN')  # Or set manually: kappa = 0.1

# Find optimal stopping point using discrepancy principle
stopping_depth = tree.get_discrepancy_stop(critical_value=kappa)

# Make predictions
X_test = np.random.randn(20, 5)
predictions = tree.predict(X_test, depth=stopping_depth)

print(f"Optimal stopping depth: {stopping_depth}")
print(f"Test predictions: {predictions[:5]}")
```

## Repository Structure

```
Early-Stopping-Regression-Trees/
├── README.md
├── requirements.txt               # Python dependencies
├── src/                          # Source code (modular organization)
│   ├── algorithms/               # Algorithm implementations
│   │   ├── global_regression_tree.py    # Global early stopping
│   │   ├── semi_global_regression_tree.py # Semi-global early stopping  
│   │   └── two_step.py                   # Two-step procedure
│   └── utils/                    # Utility functions
│       ├── data_generation.py           # Data generating processes
│       └── noise_level_estimator.py     # Noise level estimation
├── experiments/                  # Experimental code
│   ├── simulations/              # Simulation studies
│   │   ├── regression_tree_simulation.py # Clean simulation framework
│   │   └── results/              # Simulation output files
│   └── empirical_studies/        # Real data experiments
│       └── empirical_application.py     # Empirical evaluation
├── data/                         # Datasets
│   ├── Boston.csv                # Boston housing dataset
│   ├── communities.csv           # Communities and crime dataset
│   ├── ozone_clean.csv          # Ozone dataset
│   └── Servo.csv                # Servo dataset
├── results/                      # Generated results and tables
└── visualization/                # Plotting and visualization
    ├── low_dim_heatmaps.py       # Low-dimensional examples
    ├── tree_stopping_gif.py      # Animation of tree growth
    ├── create_gif_from_frames.py # GIF creation script
    ├── gif/                      # Animation frames
    └── tree_stopping_animation.gif # Generated animation
```

## Key Features

### **🚀 Performance & Efficiency**
- **Parallel processing**: Automatic CPU detection for simulations
- **Progress tracking**: Real-time progress bars with time estimates
- **Error resilience**: Robust handling of method failures
- **Fast execution**: 3-5x speedup through optimized parallel processing

### **📊 Multiple Algorithms**
- **Global Early Stopping**: Main algorithm with discrepancy principle
- **Semi-Global Early Stopping**: Efficient local criterion approach
- **Two-Step Procedure**: Hybrid depth estimation + pruning
- **Interpolation**: Smooth transitions between tree depths
- **Benchmarks**: Cost complexity pruning, deep trees, balanced oracle

### **🔬 Comprehensive Evaluation**
- **8 Data Generation Processes**: From simple 2D to complex high-dimensional
- **Statistical Summaries**: Median results with mean absolute deviation
- **Professional Output**: Clean CSV files and LaTeX-ready tables
- **Reproducible Results**: Global random seed configuration

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ratmir-miftachov/Early-Stopping-Regression-Trees.git
cd Early-Stopping-Regression-Trees
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Experiments

### Simulations
```bash
cd experiments/simulations
python regression_tree_simulation.py
```

### Empirical Studies
```bash
cd experiments/empirical_studies  
python empirical_application.py
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{miftachov2025early,
  title={Early Stopping for Regression Trees},
  author={Miftachov, Ratmir and Rei{\ss}, Markus},
  journal={arXiv preprint arXiv:2502.04709},
  year={2025}
}
```

## Contact

- **Ratmir Miftachov**: contact[at]miftachov.com
- **Markus Reiß**: reismark[at]hu-berlin.de


