# Source Code Directory

This directory contains all source code following a clean, modular structure inspired by best practices in software engineering and machine learning research.

## Directory Structure

```
src/
├── algorithms/          # Algorithm implementations
│   ├── __init__.py
│   ├── global_regression_tree.py     # Global early stopping algorithms
│   ├── semi_global_regression_tree.py # Semi-global early stopping
│   └── two_step.py                   # Two-step early stopping procedure
└── utils/              # Utility functions
    ├── __init__.py
    ├── data_generation.py            # Data generation for experiments
    └── noise_level_estimator.py      # Noise level estimation methods
```

## Module Descriptions

### **Algorithms**

#### `global_regression_tree.py`
**Main class**: `RegressionTree`

**Key methods**: 
- `iterate(max_depth)`: Build tree iteratively using CART splitting
- `get_discrepancy_stop(critical_value)`: Find early stopping point using discrepancy principle
- `get_balanced_oracle()`: Oracle stopping criterion (theoretical benchmark)
- `predict(X, depth)`: Make predictions at any tree depth
- `get_n_leaves()`: Count leaves at current depth

**Features**:
- Global residual monitoring for early stopping
- Breadth-first and best-first search algorithms
- Linear interpolation between tree depths
- Oracle inequalities without smoothness assumptions

#### `semi_global_regression_tree.py`  
**Main classes**: `DecisionTreeRegressor`, `Node`

**Key methods**:
- `iterate(max_depth)`: Iterative tree building with semi-global criterion
- `predict(X)`: Prediction with semi-global early stopping
- `take_snapshot()`: Capture tree state for analysis

**Features**:
- Local impurity-based stopping criterion
- Priority queue for node selection
- Efficient for large datasets
- Snapshot functionality for debugging

#### `two_step.py`
**Main function**: `esmp(max_depth, X, y, alpha, cv)`

**Features**:
- Combines initial depth estimation with post-pruning
- Cross-validation for alpha parameter selection
- Hybrid approach leveraging benefits of both strategies

### **Utils**

#### `data_generation.py`
**Main function**: `generate_data_from_X(X, noise_level, dgp_name, n_points, add_noise)`

**Supported DGPs**: 
- **2D Cases**: `circular`, `sine_cosine`, `rectangular`, `smooth_signal`
- **High-dimensional**: `additive_smooth`, `additive_step`, `additive_linear`, `additive_hills`
- **Classic**: `friedman1`, `breiman84`

**Features**:
- Flexible noise level control
- Multiple function classes (smooth, step, linear, hills)
- Consistent interface across all DGPs

#### `noise_level_estimator.py`
**Main class**: `Estimator`

**Methods**: 
- `estimate(method='1NN')`: 1-nearest neighbor noise estimation
- Support for multiple estimation techniques

**Features**:
- Data-driven noise level estimation
- Used by early stopping algorithms for critical value setting
- Robust to different data distributions

## Key Design Principles

### **1. Modular Organization**
- Clear separation between algorithms and utilities
- Each algorithm in its own module with focused responsibility
- Consistent naming conventions across modules
- Easy to understand and maintain

### **2. Clean Interfaces**
- Standardized method signatures across algorithms
- Consistent parameter naming (e.g., `X`, `y`, `max_depth`)
- Clear documentation and type hints
- Predictable behavior across implementations

### **3. Easy Extensibility**
- **Adding new algorithms**: Create new file in `algorithms/`
- **Adding new DGPs**: Extend `data_generation.py`
- **Adding new noise estimators**: Extend `noise_level_estimator.py`
- **Adding new utilities**: Create new files in `utils/`

## Import Structure

From experiment files, import using the new modular structure:

```python
# Algorithms
from src.algorithms import global_regression_tree as rt
from src.algorithms import semi_global_regression_tree as semi
from src.algorithms import two_step

# Utils  
from src.utils import data_generation
from src.utils import noise_level_estimator as noise_est
```

## Algorithm Comparison

| **Algorithm** | **Type** | **Speed** | **Performance** | **Use Case** |
|---------------|----------|-----------|-----------------|--------------|
| Global ES | Global criterion | Fast | Best overall | General purpose |
| Semi-global ES | Local criterion | Medium | Good | Large datasets |
| Two-step | Hybrid | Slow | Good | When depth estimation needed |

### **Performance Characteristics**

- **Global ES**: 
  - ✅ Excellent statistical performance
  - ✅ Fast execution
  - ✅ Theoretical guarantees
  - ❌ Requires noise level estimation

- **Semi-global ES**:
  - ✅ Good for large datasets
  - ✅ No noise estimation required
  - ✅ Efficient memory usage
  - ❌ Local criterion may be suboptimal

- **Two-step**:
  - ✅ Combines best of both approaches
  - ✅ Cross-validation built-in
  - ❌ Slower execution
  - ❌ More complex implementation

## Dependencies

### **Core Requirements**
- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `scikit-learn`: Tree implementations and cross-validation

### **Optional Dependencies**
- `statsmodels`: Advanced noise estimation methods
- `matplotlib`: Visualization support
- `tqdm`: Progress bars (for experiments)

## Implementation Notes

### **Memory Management**
- Efficient tree representations
- Minimal memory footprint for large trees
- Garbage collection considerations

### **Numerical Stability**
- Robust handling of edge cases
- Careful handling of floating-point arithmetic
- Proper handling of degenerate splits

### **Error Handling**
- Graceful degradation when methods fail
- Informative error messages
- Fallback strategies for edge cases

## Testing and Validation

The modular structure enables easy unit testing:

```python
# Test individual algorithms
from src.algorithms import global_regression_tree as rt
tree = rt.RegressionTree(X, y)
tree.iterate(max_depth=5)
predictions = tree.predict(X_test, depth=3)

# Test utilities
from src.utils import data_generation as dg
X, y = dg.generate_data_from_X(X, noise_level=1.0, dgp_name='circular')
```

## Future Extensions

The modular structure makes it easy to add:
- **New early stopping criteria**
- **Additional tree growing strategies** 
- **Advanced noise estimation methods**
- **Alternative splitting criteria**
- **Ensemble methods**

The clean separation of concerns ensures that **adding new functionality doesn't break existing code** and maintains the overall system architecture. 