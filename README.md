# Early Stopping for Regression Trees

This repository contains the implementation code for the paper [**"Early Stopping for Regression Trees"**](https://arxiv.org/abs/2502.04709) by Ratmir Miftachov and Markus Reiß.

## 📝 Abstract

We develop early stopping rules for growing regression tree estimators. The fully data-driven stopping rule is based on monitoring the global residual norm. The best-first search and the breadth-first search algorithms together with linear interpolation give rise to generalized projection or regularization flows. A general theory of early stopping is established. Oracle inequalities for the early-stopped regression tree are derived without any smoothness assumption on the regression function, assuming the original CART splitting rule, yet with a much broader scope. The remainder terms are of smaller order than the best achievable rates for Lipschitz functions in dimension d≥2. In real and synthetic data the early stopping regression tree estimators attain the statistical performance of cost-complexity pruning while significantly reducing computational costs.

## 🏗️ Repository Structure

```
early-stopping-cart/
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
│   └── tree_stopping_gif.py      # Animation of tree growth with early stopping
└── scripts/                       # Main runner scripts (optional)
```

## 🔧 Installation

### Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Joblib

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ratmir-miftachov/Early-Stopping-CART.git
cd Early-Stopping-CART
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Usage

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
kappa = 0.1  # Critical value
stopping_depth = tree.get_discrepancy_stop(critical_value=kappa)

# Make predictions
X_test = np.random.randn(20, 5)
predictions = tree.predict(X_test, depth=stopping_depth)

print(f"Optimal stopping depth: {stopping_depth}")
print(f"Test predictions: {predictions[:5]}")
```

### Empirical Evaluation

Run empirical analysis on real datasets:

```python
# Navigate to experiments directory
import sys
sys.path.append('experiments')
sys.path.append('src')
import empirical_application

# The empirical_application.py script contains functions for:
# - Loading real datasets (Boston, Communities, Servo, Ozone)
# - Comparing early stopping with other methods
# - Generating performance tables and visualizations
```

### Simulation Studies

Generate synthetic data and run simulations:

```python
# Navigate to src directory for data generation
import sys
sys.path.append('src')
import data_generation as dg

# Generate data from various data generating processes
X = np.random.uniform(-2.5, 2.5, (200, 4))
y, noise = dg.generate_data_from_X(X, noise_level=0.1, dgp_name='additive_smooth')

# Run simulation comparison
sys.path.append('experiments')
import simulation_MSE_tables
# This will generate MSE comparison tables for different methods
```

## 📊 Key Features

### 1. **Global Early Stopping**
- Monitor global residual norm for stopping criterion
- Discrepancy principle for automatic threshold selection
- Linear interpolation for continuous stopping

### 2. **Multiple Data Generating Processes**
- Additive smooth functions
- Step functions
- Hills functions  
- Breiman 1984 and Friedman examples
- Sine-cosine combinations

### 3. **Comprehensive Benchmarking**
- Comparison with cost-complexity pruning
- Local early stopping methods
- Semi-global approaches
- Deep tree baselines

### 4. **Real Data Applications**
- Boston housing prices
- Communities and crime data
- Servo motor data
- Ozone concentration levels

## 📈 Performance

The early stopping approach achieves:
- **Statistical performance** comparable to cost-complexity pruning
- **Significantly reduced** computational costs
- **Oracle inequalities** without smoothness assumptions
- **Minimax optimal** rates for Lipschitz functions

## 🔬 Reproduce Paper Results

To reproduce the main results from the paper:

1. **Simulation Studies:**
```bash
cd experiments
python simulation_MSE_tables.py
python simulation_rel_eff_boxplots.py
```

2. **Empirical Applications:**
```bash
cd experiments
python empirical_application.py
```

3. **Visualizations:**
```bash
cd visualization
python low_dim_heatmaps.py
python tree_stopping_gif.py
```

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{miftachov2025early,
  title={Early Stopping for Regression Trees},
  author={Miftachov, Ratmir and Rei{\ss}, Markus},
  journal={arXiv preprint arXiv:2502.04709},
  year={2025}
}
```

## 📧 Contact

- **Ratmir Miftachov**: [Email from arXiv]
- **Markus Reiß**: [Email from arXiv]

## 📜 License

This project is available under standard academic usage terms. Please cite the paper if you use this code for research purposes.

## 🔗 Links

- [arXiv Paper](https://arxiv.org/abs/2502.04709)
- [GitHub Repository](https://github.com/ratmir-miftachov/Early-Stopping-CART)
