# Clear all variables from the global namespace
for name in list(globals()):
    if not name.startswith('_'):
        del globals()[name]

# Import required libraries and modules
import numpy as np
import os
import importlib
import sys

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, KFold
import CARTRM as RM
import semiglobal as semi
import data_generation
from joblib import Parallel, delayed
import noise_level_estimator as noise_est
import es_mp
import time
import pandas as pd

import regression_tree as rt

# Reload modules to ensure changes are applied
importlib.reload(data_generation)
importlib.reload(semi)
importlib.reload(rt)

def methods_stopping(X_train, y_train, X_test, y_test, stopping, noise_level, noise, k_cv, true_signal, noise_test, true_signal_test, crit):
    """
       Execute the specified stopping method on decision trees using various criteria.

       Parameters:
       X_train, y_train (np.ndarray): Training data features and targets.
       X_test, y_test (np.ndarray): Test data features and targets.
       stopping (str): The type of stopping criterion ('local', 'global', 'pruning', 'deep', 'semi').
       noise_level (float): Noise level used in the simulation.
       noise (np.ndarray): Array of noise values.
       k_cv (KFold): Cross-validation generator.
       true_signal (np.ndarray): Array of the true signals for training set.
       noise_test (np.ndarray): Array of noise values for the test set.
       true_signal_test (np.ndarray): Array of the true signals for test set.
       crit (str): Criterion used for decision tree regression ('sigma', 'NN', 'epsilon').

       Returns:
       float: Mean squared error for the test predictions depending on the stopping method.
       """

    if crit == 'sigma':
        kappa = noise_level
    elif crit == 'NN':
        estimator = noise_est.Estimator(X_train, y_train)
        kappa = estimator.estimate(method='1NN')
    elif crit == 'epsilon':
        kappa = np.mean(noise**2)


    if stopping == 'local':
        return local_ES(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                        noise_level=noise_level, noise=noise, true_signal_test=true_signal_test, crit=crit)
    elif stopping == 'global':
        return global_ES(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                         noise_level=noise_level, noise=noise, k_cv=k_cv,
                         true_signal=true_signal, noise_test=noise_test, signal_test=true_signal_test, kappa=kappa)
    elif stopping == 'pruning':
        return pruning_tree(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, k_cv=k_cv, true_signal_test=true_signal_test)
    elif stopping == 'deep':
        return deep_tree(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, true_signal_test=true_signal_test)
    elif stopping == 'semi':
        return semi_ES(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                       noise_level=noise_level, noise=noise, true_signal=true_signal,
                       noise_test=noise_test, true_signal_test=true_signal_test, kappa=kappa)

def local_ES(X_train, y_train, X_test, y_test, noise_level, noise, true_signal_test, crit):
    """
    Apply local early stopping criteria for a decision tree regressor.

    Parameters:
    X_train, y_train (np.ndarray): Training data features and targets.
    X_test, y_test (np.ndarray): Test data features and targets.
    noise_level (float): The level of noise in the data.
    noise (np.ndarray): Array of noise values.
    true_signal_test (np.ndarray): Array of the true signals for the test set.
    crit (str): Criterion used for modifying the decision tree parameters.

    Returns:
    float: The mean squared error of the predictions on the test data.
    """
    if crit == 'sigma':
        # sigma2=noise_level**2 OR noise_vector = noise OR sigma_est_method='1NN', sigma_est_method='LS'
        tree_local = RM.DecisionTreeRegressor(max_depth=40, loss='mse', global_es=False, min_samples_split=2, sigma2=noise_level)
        tree_local.train(X_train, y_train)
    elif crit == 'NN':
        tree_local = RM.DecisionTreeRegressor(max_depth=40, loss='mse', global_es=False, min_samples_split=2, sigma_est_method='1NN')
        tree_local.train(X_train, y_train)
    elif crit == 'epsilon':
        tree_local = RM.DecisionTreeRegressor(max_depth=40, loss='mse', global_es=False, min_samples_split=2, noise_vector = noise)
        tree_local.train(X_train, y_train)

    predictions_test = tree_local.predict(X_test)
    mspe_local = np.mean((predictions_test - true_signal_test)**2)
    return mspe_local

def semi_ES(X_train, y_train, X_test, y_test, noise_level, noise, true_signal, noise_test, true_signal_test, kappa):
    """
    Apply semi-global early stopping criteria for a decision tree regressor.

    Parameters:
    X_train, y_train (np.ndarray): Training data features and targets.
    X_test, y_test (np.ndarray): Test data features and targets.
    noise_level (float): The level of noise in the data.
    noise (np.ndarray): Noise array affecting the training data.
    true_signal (np.ndarray): True signals for the training data.
    noise_test (np.ndarray): Noise array affecting the test data.
    true_signal_test (np.ndarray): True signals for the test data.
    kappa (float): Kappa value derived from noise estimation.

    Returns:
    float: The mean squared error of the predictions on the test data.
    """
    semi_time_start = time.time()
    tree_semi = semi.DecisionTreeRegressor(X_train, y_train, max_iter=3000,
                                            min_samples_split=1, kappa = kappa)
    tree_semi.iterate(max_depth=35)
    semi_time_end = time.time()
    semi_n_leaves = tree_semi.stopping_iteration
    predictions_test = tree_semi.predict(X_test)
    mspe_semi = np.mean((predictions_test - true_signal_test)**2)

    semi_time = round(semi_time_end - semi_time_start, 2)
    print("semi time is:", semi_time)
    print("semi n leaves is:", semi_n_leaves)
    return mspe_semi, semi_n_leaves, semi_time

def global_ES(X_train, y_train, X_test, y_test, noise_level, noise, k_cv, true_signal, noise_test, signal_test, kappa):
    """
    Apply global early stopping criteria based on the bias-variance tradeoff for a decision tree regressor.

    Parameters:
    X_train, y_train, X_test, y_test (np.ndarray): Training and testing data features and targets.
    noise_level (float): The level of noise in the data.
    noise (np.ndarray): Noise array affecting the data.
    k_cv (KFold): Cross-validation generator for estimating model performance.
    true_signal (np.ndarray): True signals for the training data.
    noise_test (np.ndarray): Noise array affecting the test data.
    signal_test (np.ndarray): True signals for the test data.
    kappa (float): Threshold parameter influencing the stopping criterion based on noise estimation.

    Returns:
    Tuple[float]: global ES, 2Step, Oracle, Interpolated on training set, Oracle interpolated on test set.
    """

    regression_tree = rt.RegressionTree(design=X_train, response=y_train, min_samples_split=1, true_signal=true_signal,
                            true_noise_vector=noise)
    regression_tree.iterate(max_depth=30)
    early_stopping_iteration = regression_tree.get_discrepancy_stop(critical_value=kappa)
    regression_tree.get_n_leaves()

    # Balanced oracle:
    balanced_oracle_iteration = regression_tree.get_balanced_oracle() + 1
    prediction_balanced_oracle = regression_tree.predict(X_test, depth=balanced_oracle_iteration)
    mspe_global_balanced_oracle = np.mean((prediction_balanced_oracle - signal_test) ** 2)
    # Global ES prediction
    prediction_global_k1 = regression_tree.predict(X_test, depth=early_stopping_iteration)
    mspe_global = np.mean((prediction_global_k1 - signal_test) ** 2)
    #Interpolation:
    if early_stopping_iteration == 0:
        mspe_global_inter = mspe_global
        print('No Interpolation done.')
    else:
        prediction_global_k = regression_tree.predict(X_test, depth=early_stopping_iteration-1)
        residuals = regression_tree.residuals
        r2_k1 = residuals[early_stopping_iteration]
        r2_k = residuals[early_stopping_iteration - 1]
        alpha = 1 - np.sqrt(1 - (r2_k - kappa) / (r2_k - r2_k1))
        predictions_interpolated = (1 - alpha) * prediction_global_k + alpha * prediction_global_k1
        mspe_global_inter = np.mean((predictions_interpolated - signal_test) ** 2)

    #2-Step:
    two_step_time_start = time.time()
    m = early_stopping_iteration + 1
    _ , tree_two_step, _ = es_mp.esmp(m, X_train, y_train, 0.01, k_cv)
    two_step_time_end = time.time()
    prediction_two_step = tree_two_step.predict(X_test)
    mspe_two_step = np.mean((prediction_two_step - signal_test) ** 2)
    two_step_n_leaves = tree_two_step.get_n_leaves()

    # To get n_leaves:
    global_time_start = time.time()
    regression_tree_count = rt.RegressionTree(design=X_train, response=y_train, min_samples_split=1)
    regression_tree_count.iterate(max_depth=early_stopping_iteration+1) # +1 to get the correct level. Already couble checked!
    global_time_end = time.time()
    global_n_leaves = regression_tree_count.get_n_leaves()
    global_time = round(global_time_end - global_time_start, 2)
    #print(global_time)
    regression_tree_count.iterate(max_depth=balanced_oracle_iteration+1) # +1 to get the correct level. Already couble checked!
    balanced_n_leaves = regression_tree_count.get_n_leaves()

    two_step_time = round((two_step_time_end - two_step_time_start) + global_time, 2)
    print("global n leaves:", global_n_leaves)
    print("global time:", global_time)
    # global ES, 2Step, Oracle, Interpolated on training set + leaves
    return mspe_global, mspe_two_step, mspe_global_balanced_oracle, mspe_global_inter, global_n_leaves, two_step_n_leaves, balanced_n_leaves, global_time, two_step_time

def deep_tree(X_train, y_train, X_test, y_test, true_signal_test):
    """
    Train a deep decision tree without any early stopping to act as a baseline comparison

    Parameters:
    X_train, y_train (np.ndarray): Training data features and targets.
    X_test, y_test (np.ndarray): Test data features and targets.
    true_signal_test (np.ndarray): True signals for the test data.

    Returns:
    float: The mean squared error of the deep tree predictions on the test data.
    """

    deep_tree_start = time.time()
    regression_tree = rt.RegressionTree(design=X_train, response=y_train, min_samples_split=1)
    regression_tree.iterate(max_depth=40)
    deep_tree_end = time.time()
    max_possible_depth = len(regression_tree.residuals)
    predictions_test = regression_tree.predict(X_test, depth = max_possible_depth)
    mspe_deep = np.mean((predictions_test - true_signal_test) ** 2)

    deep_time = round(deep_tree_end - deep_tree_start, 2)

    return mspe_deep, deep_time

def pruning_tree(X_train, y_train, X_test, y_test, k_cv, true_signal_test):
    """
    Apply cost complexity pruning to a decision tree and evaluate its performance.

    Parameters:
    X_train, y_train, X_test, y_test (np.ndarray): Training and testing data features and targets.
    k_cv (KFold): Cross-validation generator for determining the best pruning level.
    true_signal_test (np.ndarray): True signals for the test data.

    Returns:
    Tuple[float, float]: MSE after pruning and MSE using the oracle selected pruning level.
    """
    # Pruning on training set:
    pruning_start = time.time()
    tree_pruning = DecisionTreeRegressor(max_depth=40)
    tree_pruning.fit(X_train, y_train)
    path = tree_pruning.cost_complexity_pruning_path(X_train, y_train)
    alpha_sequence, impurities = path.ccp_alphas[1:-1], path.impurities[1:-1]
    threshold = 0.0005 
    filtered_alpha_sequence = np.array([alpha_sequence[0]])
    filtered_impurities = np.array([impurities[0]])  # Include the first impurity
    for u in range(1, len(alpha_sequence)):
        impurity_change = impurities[u] - impurities[u - 1]
        if impurity_change >= threshold:
            filtered_alpha_sequence = np.append(filtered_alpha_sequence, alpha_sequence[u])
            filtered_impurities = np.append(filtered_impurities, impurities[u])
    trees = []
    for ccp_alpha in filtered_alpha_sequence:
        dtree_alpha = DecisionTreeRegressor(ccp_alpha=ccp_alpha)
        dtree_alpha.fit(X_train, y_train)
        trees.append(dtree_alpha)

    parameters = {'ccp_alpha': filtered_alpha_sequence.tolist()}
    gsearch = GridSearchCV(DecisionTreeRegressor(), parameters, cv=k_cv, scoring='neg_mean_squared_error')
    gsearch.fit(X_train, y_train)
    clf = gsearch.best_estimator_
    pruning_end = time.time()
    predictions_test_pruning = clf.predict(X_test)
    mspe_pruning = np.mean((predictions_test_pruning - true_signal_test) ** 2)

    pruning_n_leaves = clf.get_n_leaves()
    pruning_time = round(pruning_end - pruning_start, 2)

    return mspe_pruning, pruning_n_leaves, pruning_time

def run_simulation(dgp, M=300, n_points=63, noise_level=1, crit='sigma'): # crit = 'sigma', 'NN', 'epsilon'
    """
       Run simulation for a specified data generating process (DGP) with multiple configurations.

       Parameters:
       dgp (str): Name of the data generating process.
       M (int): Number of Monte Carlo simulation runs.
       n_points (int): Number of data points per run.
       noise_level (float): Standard deviation of the noise added to the data.
       crit (str): Criterion used for noise estimation ('sigma', 'NN', 'epsilon').

       Returns:
       np.ndarray: Array of results from different stopping criteria applied in decision trees.
       """
    if crit == None:
        print('Please specify the kappa threshold')

    all_X_train, all_y_train, all_noise_train, all_f, all_f_test = [], [], [], [], []
    all_X_test, all_y_test, all_noise_test = [], [], []

    for _ in range(M):
        # X deterministic:
        if dgp in ['sine_cosine', 'rectangular', 'circular', 'smooth_signal']:
            n_train = 1000
            d = 5 
            X_train = np.random.uniform(0, 1, size=(n_train, d))
            X_test = np.random.uniform(0, 1, size=(n_train, d))

        elif dgp == 'additive_smooth' or dgp == 'additive_step' or 'additive_linear' or 'additive_hills':
            n_train = 1000
            n_test = 1000
            d = 30 
            X_train = np.random.uniform(-2.5, 2.5, size=(n_train, d))
            X_test = np.random.uniform(-2.5, 2.5, size=(n_test, d))
        # X random:
        elif dgp == 'breiman84':
            n_train = 500
            n_test = 500
            X1_train = np.random.choice([-1, 1], size=n_train, p=[0.5, 0.5])
            X2_10_train = np.random.choice([-1, 0, 1], size=(n_train, 9), p=[1 / 3, 1 / 3, 1 / 3])
            X_train = np.column_stack((X1_train, X2_10_train))

            # Generate on test set:
            X1_test = np.random.choice([-1, 1], size=n_test, p=[0.5, 0.5])
            X2_10_test = np.random.choice([-1, 0, 1], size=(n_test, 9), p=[1 / 3, 1 / 3, 1 / 3])
            X_test = np.column_stack((X1_test, X2_10_test))
        # X random:
        elif dgp == 'friedman1':
            n_train = 500
            n_test = 500
            d = 10
            X_train = np.random.uniform(0, 1, size=(n_train, d))
            X_test = np.random.uniform(0, 1, size=(n_test, d))

        y_train, noise_train = data_generation.generate_data_from_X(X_train, noise_level, dgp_name=dgp, n_points=n_points, add_noise=True)
        y_test, noise_test = data_generation.generate_data_from_X(X_test, noise_level, dgp_name=dgp, n_points=n_points, add_noise=True) # TODO: That is why n_test=n_train required
        f, nuisance = data_generation.generate_data_from_X(X_train, noise_level, dgp_name=dgp, n_points=n_points, add_noise=False)
        f_test, nuisance = data_generation.generate_data_from_X(X_test, noise_level, dgp_name=dgp, n_points=n_points, add_noise=False) # TODO: That is why n_test=n_train required

        all_f.append(f)
        all_f_test.append(f_test)

        all_X_train.append(X_train)
        all_y_train.append(y_train)
        all_noise_train.append(noise_train)

        all_X_test.append(X_test)
        all_y_test.append(y_test)
        all_noise_test.append(noise_test)

    def monte_carlo(stopping_method, k_cv):

        mspe_list = []
        additional_metric_list = []
        additional_metric2_list = []
        additional_metric3_list = []
        additional_metric4_list = []
        additional_metric5_list = []
        additional_metric6_list = []
        additional_metric7_list = []
        additional_metric8_list = []
        additional_metric9_list = []


        for i in range(M):
            print(dgp, stopping_method, i)
            k_cv_iter = k_cv[i]

            results = methods_stopping(all_X_train[i], all_y_train[i], all_X_test[i], all_y_test[i],
                                               stopping=stopping_method, noise_level=noise_level,
                                               noise=all_noise_train[i], k_cv=k_cv_iter,
                                       true_signal=all_f[i], noise_test=all_noise_test[i],
                                       true_signal_test = all_f_test[i], crit=crit)
            if not isinstance(results, tuple):
                results = (results,)
            mspe = results[0]
            mspe_list.append(mspe)

            # Handle the additional metric if it exists
            if len(results) > 1:
                additional_metric_list.append(results[1])
            if len(results) > 2:
                additional_metric2_list.append(results[2])
            if len(results) > 3:
                additional_metric3_list.append(results[3])
            if len(results) > 4:
                additional_metric4_list.append(results[4])
            if len(results) > 5:
                additional_metric5_list.append(results[5])
            if len(results) > 6:
                additional_metric6_list.append(results[6])
            if len(results) > 7:
                additional_metric7_list.append(results[7])
            if len(results) > 8:
                additional_metric8_list.append(results[8])
            if len(results) > 9:
                additional_metric9_list.append(results[9])

        mean_mspe = np.array(mspe_list)
        mean_additional_metric = np.array(additional_metric_list) if additional_metric_list else None
        mean_additional_metric2 = np.array(additional_metric2_list) if additional_metric2_list else None
        mean_additional_metric3 = np.array(additional_metric3_list) if additional_metric3_list else None
        mean_additional_metric4 = np.array(additional_metric4_list) if additional_metric4_list else None
        mean_additional_metric5 = np.array(additional_metric5_list) if additional_metric5_list else None
        mean_additional_metric6 = np.array(additional_metric6_list) if additional_metric6_list else None
        mean_additional_metric7 = np.array(additional_metric7_list) if additional_metric7_list else None
        mean_additional_metric8 = np.array(additional_metric8_list) if additional_metric8_list else None
        mean_additional_metric9 = np.array(additional_metric9_list) if additional_metric9_list else None


        return (mean_mspe, mean_additional_metric, mean_additional_metric2, mean_additional_metric3,
                mean_additional_metric4, mean_additional_metric5, mean_additional_metric6, mean_additional_metric7, mean_additional_metric8,
                mean_additional_metric9)

    cv_splits = [KFold(n_splits=5, shuffle=True, random_state=42 + i) for i in range(M)]

    #mspe_local_mean = monte_carlo('local', k_cv=cv_splits)[0]
    mspe_semi_mean, semi_n_leaves, semi_time = monte_carlo('semi', k_cv=cv_splits)[0:3]

    (mspe_global_mean, mspe_global_pruning_mean, mspe_global_balanced_oracle, mspe_global_inter,
     global_n_leaves, two_step_n_leaves, balanced_n_leaves, global_time, two_step_time) = monte_carlo(stopping_method='global', k_cv=cv_splits)[0:9]

    mspe_pruning_mean, pruning_n_leaves, pruning_time = monte_carlo('pruning', k_cv=cv_splits)[0:3]
    mspe_deep_mean, deep_time = monte_carlo('deep', k_cv=cv_splits)[0:2]

    return np.column_stack((mspe_deep_mean, mspe_pruning_mean,
                            mspe_global_pruning_mean, mspe_global_mean, mspe_global_inter,
                            mspe_global_balanced_oracle, mspe_semi_mean,
                            pruning_n_leaves, global_n_leaves, global_n_leaves, two_step_n_leaves, semi_n_leaves,
                            pruning_time, global_time, global_time, two_step_time, semi_time, deep_time))

def run_simulation_wrapper(dgp_name):

    if dgp_name == 'smooth_signal':
        return run_simulation(dgp_name, noise_level=1)
    elif dgp_name == 'breiman84':
        return run_simulation(dgp_name, noise_level=1)
    return run_simulation(dgp_name)

def main():
    """
      Main function to run simulations for specified DGPs, compute statistics, and generate plots.
      Utilizes parallel processing for efficiency.
      """
    dgps = ['circular', 'sine_cosine', 'rectangular', 'smooth_signal', 'additive_smooth', 'additive_step', 'additive_linear', 'additive_hills']
    results = Parallel(n_jobs=8)(delayed(run_simulation_wrapper)(dgp) for dgp in dgps)
    results = np.array(results)
    results_sqrt = np.round(np.sqrt(results[:, :, 0:7]), 3)

    median = np.median(results_sqrt, axis=1)
    deviation = np.mean(np.abs(results_sqrt - median[:, np.newaxis, :]), axis=1)

    column_names = ["Deep", "Pruning", "Two-Step", "Global", "Global Inter", "Bal Oracle", "Semi"]

    row_names =['Circular', 'Sine Cosine', 'Rectangular', 'Smooth Signal',
                 'Additive Smooth', 'Additive Step', 'Additive Linear', 'Additive Hills']
    #dir = '/Users/ratmir/PycharmProjects/RandomForest'
    dir = '/home/RDC/miftachr/H:/miftachr/ES_CART/boxplots_sigma'

    print(simple_latex_table(median, deviation, column_names, row_names, name_threshold='median_sigma', table_dir=dir))

    # Number of terminal nodes:
    n_leaves_array = results[:, :, 7:12]
    median_n_leaves = np.median(n_leaves_array, axis=1)
    df = pd.DataFrame(median_n_leaves, index=row_names,
                      columns=['Pruning', 'Global', 'Global Inter',  'Two-step', 'Semi'])
    print(df)

    # Run times:
    run_times = results[:, :, 12:18]
    run_times_median = np.median(run_times, axis=1)
    run_times_deviation =  np.mean(np.abs(run_times - run_times_median[:, np.newaxis, :]), axis=1)
    print(simple_latex_table(run_times_median, run_times_deviation, column_names=["Pruning", "Global", "Global Inter",
                             "Two-Step", "Semi", "Deep"],
                             row_names=row_names, name_threshold='median_sigma_times', table_dir=dir))



def simple_latex_table(arr1, arr2, column_names, row_names, name_threshold, table_dir=None):

    # Start the LaTeX table with minimal formatting
    latex_str = "\\begin{tabular}{l" + "c" * 8 + "}\n\\hline\n"  # Adjusted for 8 columns

    # Add column headers
    latex_str += " & " + " & ".join(column_names) + " \\\\ \\hline\n"

    # Add each row with its corresponding row name
    for name, row1, row2 in zip(row_names, arr1, arr2):
        # Formatting rows with data from both arrays
        row_str = name + " & "
        row_str += " & ".join(f"{x:.2f} ({y:.2f})" for x, y in zip(row1, row2))
        row_str += " \\\\\n"
        latex_str += row_str

    # Close the table
    latex_str += "\\hline\n\\end{tabular}"

    if table_dir is not None:
        with open(os.path.join(table_dir, f"table_{name_threshold}.txt"), "w") as f:
            f.write(latex_str)
    return latex_str



if __name__ == "__main__":
    main()

