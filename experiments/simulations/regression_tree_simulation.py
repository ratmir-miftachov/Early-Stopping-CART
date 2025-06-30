import pandas as pd
import numpy as np
import os
import sys
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, KFold
from joblib import Parallel, delayed, cpu_count
import time
from typing import Dict, List, Any

# Optional progress bar - install with: pip install tqdm
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.algorithms import semi_global_regression_tree as semi
from src.utils import data_generation
from src.utils import noise_level_estimator as noise_est
from src.algorithms import two_step
from src.algorithms import global_regression_tree as rt

# Global configuration
RANDOM_SEED = 42
N_MONTE_CARLO_RUNS = 2  # Change to 2 for testing

print(f"Using {cpu_count()} CPU cores for parallel processing")

def generate_simulated_data(dgp_name, n_samples, noise_level, random_state=42):
    """
    Generate simulated regression data using the data_generation module
    
    Returns X_train, X_test, y_train, y_test, f_train, f_test, noise_train, noise_test
    """
    np.random.seed(random_state)
    
    # Determine data generation parameters based on DGP
    if dgp_name in ['sine_cosine', 'rectangular', 'circular', 'smooth_signal']:
        n_train = 1000
        d = 5 
        X_train = np.random.uniform(0, 1, size=(n_train, d))
        X_test = np.random.uniform(0, 1, size=(n_train, d))
    elif dgp_name in ['additive_smooth', 'additive_step', 'additive_linear', 'additive_hills']:
        n_train = 1000
        n_test = 1000
        d = 30 
        X_train = np.random.uniform(-2.5, 2.5, size=(n_train, d))
        X_test = np.random.uniform(-2.5, 2.5, size=(n_test, d))
    else:
        raise ValueError(f"Unknown DGP: {dgp_name}")

    # Generate data
    y_train, noise_train = data_generation.generate_data_from_X(
        X_train, noise_level, dgp_name=dgp_name, n_points=63, add_noise=True)
    y_test, noise_test = data_generation.generate_data_from_X(
        X_test, noise_level, dgp_name=dgp_name, n_points=63, add_noise=True)
    f_train, _ = data_generation.generate_data_from_X(
        X_train, noise_level, dgp_name=dgp_name, n_points=63, add_noise=False)
    f_test, _ = data_generation.generate_data_from_X(
        X_test, noise_level, dgp_name=dgp_name, n_points=63, add_noise=False)
    
    return X_train, X_test, y_train, y_test, f_train, f_test, noise_train, noise_test

def run_single_iteration(seed, dgp_config):
    """Run a single Monte Carlo iteration for all stopping methods"""
    # Generate data
    X_train, X_test, y_train, y_test, f_train, f_test, noise_train, noise_test = generate_simulated_data(
        dgp_name=dgp_config['dgp_name'],
        n_samples=dgp_config['n_samples'],
        noise_level=dgp_config['noise_level'],
        random_state=seed
    )
    
    # Create cross-validation object
    k_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED + seed)
    
    # Dictionary to store results for this iteration
    iteration_results = []
    
    # Noise level estimation
    crit = dgp_config.get('crit', 'sigma')
    if crit == 'sigma':
        kappa = dgp_config['noise_level']
    elif crit == 'NN':
        estimator = noise_est.Estimator(X_train, y_train)
        kappa = estimator.estimate(method='1NN')
    elif crit == 'epsilon':
        kappa = np.mean(noise_train**2)
    
    # Semi-global early stopping
    try:
        start_time = time.time()
        tree_semi = semi.DecisionTreeRegressor(X_train, y_train, max_iter=3000,
                                               min_samples_split=1, kappa=kappa)
        tree_semi.iterate(max_depth=35)
        semi_time = time.time() - start_time
        predictions_test = tree_semi.predict(X_test)
        mspe_semi = np.mean((predictions_test - f_test)**2)
        
        iteration_results.append({
            "method": "Semi",
            "test_rmse": np.sqrt(mspe_semi),
            "n_leaves": tree_semi.stopping_iteration,
            "fit_time": semi_time,
        })
    except Exception as e:
        iteration_results.append({
            "method": "Semi",
            "test_rmse": np.nan,
            "n_leaves": np.nan,
            "fit_time": np.nan,
        })
    
    # Global early stopping
    try:
        start_time = time.time()
        regression_tree = rt.RegressionTree(design=X_train, response=y_train, 
                                           min_samples_split=1, true_signal=f_train,
                                           true_noise_vector=noise_train)
        regression_tree.iterate(max_depth=30)
        early_stopping_iteration = regression_tree.get_discrepancy_stop(critical_value=kappa)
        
        # Balanced oracle
        balanced_oracle_iteration = regression_tree.get_balanced_oracle() + 1
        prediction_balanced_oracle = regression_tree.predict(X_test, depth=balanced_oracle_iteration)
        mspe_global_balanced_oracle = np.mean((prediction_balanced_oracle - f_test) ** 2)
        
        # Global ES prediction
        prediction_global_k1 = regression_tree.predict(X_test, depth=early_stopping_iteration)
        mspe_global = np.mean((prediction_global_k1 - f_test) ** 2)
        
        # Interpolation
        if early_stopping_iteration == 0:
            mspe_global_inter = mspe_global
        else:
            prediction_global_k = regression_tree.predict(X_test, depth=early_stopping_iteration-1)
            residuals = regression_tree.residuals
            r2_k1 = residuals[early_stopping_iteration]
            r2_k = residuals[early_stopping_iteration - 1]
            alpha = 1 - np.sqrt(1 - (r2_k - kappa) / (r2_k - r2_k1))
            predictions_interpolated = (1 - alpha) * prediction_global_k + alpha * prediction_global_k1
            mspe_global_inter = np.mean((predictions_interpolated - f_test) ** 2)
        
        # Two-step
        m = early_stopping_iteration + 1
        _, tree_two_step, _ = two_step.esmp(m, X_train, y_train, 0.01, k_cv)
        prediction_two_step = tree_two_step.predict(X_test)
        mspe_two_step = np.mean((prediction_two_step - f_test) ** 2)
        
        # Get number of leaves
        regression_tree_count = rt.RegressionTree(design=X_train, response=y_train, min_samples_split=1)
        regression_tree_count.iterate(max_depth=early_stopping_iteration+1)
        global_n_leaves = regression_tree_count.get_n_leaves()
        two_step_n_leaves = tree_two_step.get_n_leaves()
        
        global_time = time.time() - start_time
        
        iteration_results.extend([
            {
                "method": "Global",
                "test_rmse": np.sqrt(mspe_global),
                "n_leaves": global_n_leaves,
                "fit_time": global_time,
            },
            {
                "method": "Global_Interp",
                "test_rmse": np.sqrt(mspe_global_inter),
                "n_leaves": global_n_leaves,
                "fit_time": global_time,
            },
            {
                "method": "Two_Step",
                "test_rmse": np.sqrt(mspe_two_step),
                "n_leaves": two_step_n_leaves,
                "fit_time": global_time,
            },
            {
                "method": "Bal_Oracle",
                "test_rmse": np.sqrt(mspe_global_balanced_oracle),
                "n_leaves": global_n_leaves,
                "fit_time": global_time,
            }
        ])
    except Exception as e:
        for method in ["Global", "Global_Interp", "Two_Step", "Bal_Oracle"]:
            iteration_results.append({
                "method": method,
                "test_rmse": np.nan,
                "n_leaves": np.nan,
                "fit_time": np.nan,
            })
    
    # Pruning
    try:
        start_time = time.time()
        tree_pruning = DecisionTreeRegressor(max_depth=40)
        tree_pruning.fit(X_train, y_train)
        path = tree_pruning.cost_complexity_pruning_path(X_train, y_train)
        alpha_sequence, impurities = path.ccp_alphas[1:-1], path.impurities[1:-1]
        
        # Filter alphas
        threshold = 0.0005 
        filtered_alpha_sequence = np.array([alpha_sequence[0]])
        for u in range(1, len(alpha_sequence)):
            impurity_change = impurities[u] - impurities[u - 1]
            if impurity_change >= threshold:
                filtered_alpha_sequence = np.append(filtered_alpha_sequence, alpha_sequence[u])
        
        parameters = {'ccp_alpha': filtered_alpha_sequence.tolist()}
        gsearch = GridSearchCV(DecisionTreeRegressor(), parameters, cv=k_cv, scoring='neg_mean_squared_error')
        gsearch.fit(X_train, y_train)
        clf = gsearch.best_estimator_
        pruning_time = time.time() - start_time
        
        predictions_test_pruning = clf.predict(X_test)
        mspe_pruning = np.mean((predictions_test_pruning - f_test) ** 2)
        
        iteration_results.append({
            "method": "Pruning",
            "test_rmse": np.sqrt(mspe_pruning),
            "n_leaves": clf.get_n_leaves(),
            "fit_time": pruning_time,
        })
    except Exception as e:
        iteration_results.append({
            "method": "Pruning",
            "test_rmse": np.nan,
            "n_leaves": np.nan,
            "fit_time": np.nan,
        })
    
    # Deep tree
    try:
        start_time = time.time()
        regression_tree = rt.RegressionTree(design=X_train, response=y_train, min_samples_split=1)
        regression_tree.iterate(max_depth=40)
        max_possible_depth = len(regression_tree.residuals)
        predictions_test = regression_tree.predict(X_test, depth=max_possible_depth)
        mspe_deep = np.mean((predictions_test - f_test) ** 2)
        deep_time = time.time() - start_time
        
        iteration_results.append({
            "method": "Deep",
            "test_rmse": np.sqrt(mspe_deep),
            "n_leaves": regression_tree.get_n_leaves(),
            "fit_time": deep_time,
        })
    except Exception as e:
        iteration_results.append({
            "method": "Deep",
            "test_rmse": np.nan,
            "n_leaves": np.nan,
            "fit_time": np.nan,
        })
    
    return iteration_results

def run_single_iteration_with_progress(seed, dgp_config):
    """Wrapper to add progress logging"""
    result = run_single_iteration(seed, dgp_config)
    
    # Show progress every 50 iterations (only for large iteration counts)
    if N_MONTE_CARLO_RUNS >= 50 and (seed + 1) % 50 == 0:
        print(f"Completed {seed + 1}/{N_MONTE_CARLO_RUNS} MC iterations")
    
    return result

# Mapping from DGP names to display names (in desired order)
dgp_display_names = {
    "circular": "Circular",
    "sine_cosine": "Sine Cosine", 
    "rectangular": "Rectangular",
    "smooth_signal": "Smooth Signal",
    "additive_smooth": "Additive Smooth",
    "additive_step": "Additive Step",
    "additive_linear": "Additive Linear",
    "additive_hills": "Additive Hills",
}

# List of DGP configurations to process (in desired order)
dgp_configs = [
    # 2D Cases
    {
        "dgp_name": "circular",
        "n_samples": 1000,
        "noise_level": 1.0,
        "crit": "sigma",
    },
    {
        "dgp_name": "sine_cosine", 
        "n_samples": 1000,
        "noise_level": 1.0,
        "crit": "sigma",
    },
    {
        "dgp_name": "rectangular",
        "n_samples": 1000,
        "noise_level": 1.0,
        "crit": "sigma",
    },
    {
        "dgp_name": "smooth_signal",
        "n_samples": 1000,
        "noise_level": 1.0,
        "crit": "sigma",
    },
    # Additive Models
    {
        "dgp_name": "additive_smooth",
        "n_samples": 1000,
        "noise_level": 1.0,
        "crit": "sigma",
    },
    {
        "dgp_name": "additive_step",
        "n_samples": 1000,
        "noise_level": 1.0,
        "crit": "sigma",
    },
    {
        "dgp_name": "additive_linear",
        "n_samples": 1000,
        "noise_level": 1.0,
        "crit": "sigma",
    },
    {
        "dgp_name": "additive_hills",
        "n_samples": 1000,
        "noise_level": 1.0,
        "crit": "sigma",
    },
]

def main():
    """Main function to run simulations and generate results"""
    
    # Store all results
    all_results = []
    
    # Process each DGP configuration
    for dgp_config in dgp_configs:
        dgp_name = dgp_config['dgp_name']
        print(f"\nProcessing DGP: {dgp_name}")
        
        # Run MC iterations in parallel
        n_jobs = max(1, cpu_count() - 1)
        print(f"Using {n_jobs} parallel jobs")
        
        if HAS_TQDM:
            results = Parallel(n_jobs=n_jobs)(
                delayed(run_single_iteration)(seed, dgp_config) 
                for seed in tqdm(range(N_MONTE_CARLO_RUNS), desc=f"Running {dgp_name}")
            )
        else:
            results = Parallel(n_jobs=n_jobs)(
                delayed(run_single_iteration_with_progress)(seed, dgp_config) 
                for seed in range(N_MONTE_CARLO_RUNS)
            )
        
        # Flatten results list
        dgp_results = [item for sublist in results for item in sublist]
        
        # Add dataset name to results
        display_name = dgp_display_names.get(dgp_name, dgp_name)
        for result in dgp_results:
            result["dataset"] = display_name
        
        all_results.extend(dgp_results)
    
    # Convert all results to DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Calculate median results grouped by dataset and method
    median_results = (
        df_results.groupby(["dataset", "method"])
        .agg({
            "test_rmse": ["median", lambda x: np.mean(np.abs(x - np.median(x)))],
            "n_leaves": "median",
            "fit_time": "median",
        })
        .round(3)
        .reset_index()
    )
    
    # Flatten column names
    median_results.columns = [
        "dataset", "method", "test_rmse_median", "test_rmse_mad", 
        "n_leaves_median", "fit_time_median"
    ]
    
    # Create output directory
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    median_results.to_csv(
        os.path.join(output_dir, "regression_tree_simulation.csv"),
        index=False,
    )
    
    # Print summary table
    print("\n" + "="*80)
    print("SIMULATION RESULTS SUMMARY")
    print("="*80)
    print(median_results.pivot_table(
        index='dataset', 
        columns='method', 
        values='test_rmse_median'
    ).round(3))
    
    print(f"\nResults saved to: {os.path.join(output_dir, 'regression_tree_simulation.csv')}")

if __name__ == "__main__":
    main() 