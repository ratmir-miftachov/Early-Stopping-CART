# Clear all variables from the global namespace
for name in list(globals()):
    if not name.startswith('_'):
        del globals()[name]

# Add src directory to path for imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import required libraries and modules
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, KFold
import numpy as np
import semiglobal as semi
import importlib
import time
import pandas as pd
import os
import es_mp
from joblib import Parallel, delayed
import load_dataset
import regression_tree as rt

# Reload modules to ensure changes are applied

importlib.reload(es_mp)
importlib.reload(semi)
importlib.reload(load_dataset)
importlib.reload(rt)

def empirical_simulation(i, train, test, kappa, threshold):
    print(i)
    #train and test set:
    X_train = train[i].drop(train[i].columns[0], axis=1)
    y_train = train[i].iloc[:, 0]
    X_test = test[i].drop(test[i].columns[0], axis=1)
    y_test = test[i].iloc[:, 0]

    kappa = kappa[i]

    # Global and deep tree:
    deep_tree_start = time.time()
    regression_tree = rt.RegressionTree(design=X_train, response=y_train, min_samples_split=1)
    regression_tree.iterate(max_depth=10)
    deep_tree_end = time.time()

    early_stopping_iteration = regression_tree.get_discrepancy_stop(critical_value=kappa)
    prediction_global_k1 = regression_tree.predict(X_test, depth=early_stopping_iteration)
    mspe_global = np.mean((prediction_global_k1 - y_test) ** 2)

    # Interpolation:
    if early_stopping_iteration == 0:
        mspe_global_inter = mspe_global
        print('No Interpolation done.')
    else:
        prediction_global_k = regression_tree.predict(X_test, depth=early_stopping_iteration - 1)
        residuals = regression_tree.residuals
        r2_k1 = residuals[early_stopping_iteration]
        r2_k = residuals[early_stopping_iteration - 1]
        alpha = 1 - np.sqrt(1 - (r2_k - kappa) / (r2_k - r2_k1))
        predictions_interpolated = (1 - alpha) * prediction_global_k + alpha * prediction_global_k1
        mspe_global_inter = np.mean((predictions_interpolated - y_test) ** 2)
    # Deep tree
    possible_depth = len(regression_tree.residuals)
    prediction_deep_tree = regression_tree.predict(X_test, depth=possible_depth)
    mspe_deep = np.mean((prediction_deep_tree - y_test) ** 2)
    # 2-Step:
    stop = early_stopping_iteration + 2 
    k_cv = KFold(n_splits=5, shuffle=True, random_state=42+i)
    step_two_start = time.time()
    _ , tree_two_step, filtered_alpha_sequence = es_mp.esmp(stop, X_train, y_train, threshold , k_cv=k_cv)
    step_two_end = time.time()
    prediction_two_step = tree_two_step.predict(X_test)
    mspe_two_step = np.mean((prediction_two_step - y_test)**2)
    two_step_n_leaves = tree_two_step.get_n_leaves()
    # semi-global:
    semi_es_start = time.time()

    tree_semi = semi.DecisionTreeRegressor(X_train, y_train, max_iter=3000,
                                            min_samples_split=1, kappa = kappa)
    tree_semi.iterate(max_depth=40)

    semi_es_end = time.time()
    predictions_test_semi = tree_semi.predict(X_test)
    mspe_semi = np.mean((predictions_test_semi - y_test)**2)
    semi_n_leaves = tree_semi.stopping_iteration

    #Local ES:
    local_es_start = time.time()
    tree_local = RM.DecisionTreeRegressor(max_depth=40, loss='mse', global_es=False, min_samples_split=1, sigma_est_method='1NN')
    tree_local.train(X_train, y_train)
    local_es_end = time.time()
    predictions_local = tree_local.predict(X_test)
    mspe_local = np.mean((predictions_local - y_test)**2)

    #Pruning:
    pruning_start = time.time()
    tree_pruning = DecisionTreeRegressor(max_depth=40)
    tree_pruning.fit(X_train, y_train)
    path = tree_pruning.cost_complexity_pruning_path(X_train, y_train)
    alpha_sequence, impurities = path.ccp_alphas[1:-1], path.impurities[1:-1]
    filtered_alpha_sequence = np.array([alpha_sequence[0]])  # Always include the first alpha
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
    mspe_pruning = np.mean((predictions_test_pruning - y_test)**2)
    pruning_n_leaves = clf.get_n_leaves()

    # To get n_leaves:
    global_es_start = time.time()
    regression_tree_count = rt.RegressionTree(design=X_train, response=y_train, min_samples_split=1)
    regression_tree_count.iterate(max_depth=early_stopping_iteration+1) # +1 to get the correct level. 
    global_es_end = time.time()
    global_n_leaves = regression_tree_count.get_n_leaves()

    # n_leaves:
    n_leaves_all = [pruning_n_leaves, global_n_leaves, two_step_n_leaves, semi_n_leaves]
    print(f"number of leaves are", n_leaves_all)

    print("mspe is:", round(np.sqrt(mspe_pruning),3), round(np.sqrt(mspe_two_step),3), round(np.sqrt(mspe_global),3), round(np.sqrt(mspe_global_inter),3))

    times = [
        round(deep_tree_end - deep_tree_start, 2),
        round(local_es_end - local_es_start, 2),
        round(pruning_end - pruning_start, 2),
        round(step_two_end - step_two_start + (global_es_end - global_es_start), 2),
        round(global_es_end - global_es_start, 2),
        round(global_es_end - global_es_start, 2),
        round(semi_es_end - semi_es_start, 2)
    ]
    print(f"times in seconds are:", times)
    return np.column_stack((mspe_deep, mspe_local, mspe_pruning, mspe_two_step, mspe_global, mspe_global_inter, mspe_semi)), np.array([times]), np.array([n_leaves_all])

def create_and_save_boxplot(data, dgp_name):
    plt.figure(figsize=(10, 6))
    bp = plt.boxplot(data, patch_artist=True, labels=['Deep', 'Local', 'Pruning','2 Step', 'Global', 'Global int', 'Semi'])
    # Color adjustments for box face and border
    colors = ['lightblue', 'lightyellow', 'violet', 'lightpink', 'lightgrey', 'lightgrey']


    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_linewidth(1.5)  # Set the border thickness

    # Making whiskers, caps, and medians thicker and grey
    for whisker in bp['whiskers']:
        whisker.set_linewidth(1.5)
    for cap in bp['caps']:
        cap.set_linewidth(1.5)
    for median in bp['medians']:
        median.set_linewidth(1.5)


    plt.axhline(y=1, color='black', linestyle='--', linewidth=1.5)
    plt.ylim(0, 2)

    #plt.title(f'Boxplot for {dgp_name}', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)  # Adjust the size as needed
    plt.tight_layout()  # Adjust layout
    plt.savefig(f'{dgp_name}_boxplot.png')
    plt.close()

def simple_latex_table(arr1, arr2, column_names, row_names, name_threshold, table_dir=None):

    # Start the LaTeX table with minimal formatting
    latex_str = "\\begin{tabular}{l" + "c" * 7 + "}\n\\hline\n"  # Adjusted for 7 columns

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

def main():
    #load datasets, First column: Y, the rest: X
    d_boston = load_dataset.data_boston()
    d_communities = load_dataset.data_communities()
    d_abalone = load_dataset.data_abalone()
    d_ozone = load_dataset.data_ozone()
    d_forest = load_dataset.data_forest()
    d_infrared = load_dataset.data_infrared()

    K = 500 # Number of empirical 'simulation' runs

    print('boston:')
    train, test, sigma, sigma_NN = load_dataset.split_and_sigma(d_boston, K=K)
    results_boston = Parallel(n_jobs=48)(delayed(empirical_simulation)(i, train=train, test=test, kappa=sigma_NN, threshold=0.001) for i in range(K))
    results_boston, times_boston, n_leaves_boston = zip(*results_boston)

    print('communities:')
    train, test, sigma, sigma_NN = load_dataset.split_and_sigma(d_communities, K=K)
    results_communities = Parallel(n_jobs=48)(delayed(empirical_simulation)(i, train=train, test=test, kappa=sigma_NN, threshold=0.00001) for i in range(K))
    results_communities, times_communities, n_leaves_communities = zip(*results_communities)

    print('abalone:')
    train, test, sigma, sigma_NN = load_dataset.split_and_sigma(d_abalone, K=K) # 0.005 before
    results_abalone = Parallel(n_jobs=48)(delayed(empirical_simulation)(i, train=train, test=test, kappa=sigma_NN, threshold=0.001) for i in range(K))
    results_abalone, times_abalone, n_leaves_abalone = zip(*results_abalone)

    print('ozone:')
    train, test, sigma, sigma_NN = load_dataset.split_and_sigma(d_ozone, K=K)
    results_ozone = Parallel(n_jobs=48)(delayed(empirical_simulation)(i, train=train, test=test, kappa=sigma_NN, threshold=0.001) for i in range(K))
    results_ozone, times_ozone, n_leaves_ozone = zip(*results_ozone)

    print('forest:')
    train, test, sigma, sigma_NN = load_dataset.split_and_sigma(d_forest, K=K)
    results_forest = Parallel(n_jobs=48)(delayed(empirical_simulation)(i, train=train, test=test, kappa=sigma_NN, threshold=0.001) for i in range(K))
    results_forest, times_forest, n_leaves_forest = zip(*results_forest)

    print('infrared:')
    train, test, sigma, sigma_NN = load_dataset.split_and_sigma(d_infrared, K=K)
    results_infrared = Parallel(n_jobs=48)(delayed(empirical_simulation)(i, train=train, test=test, kappa=sigma_NN, threshold=0.00003) for i in range(K))
    results_infrared, times_infrared, n_leaves_infrared = zip(*results_infrared)

    # RMSE:
    dgps_arrays = [np.concatenate(results_boston, axis=0), np.concatenate(results_communities, axis=0), np.concatenate(results_abalone, axis=0),
                   np.concatenate(results_ozone, axis=0), np.concatenate(results_forest, axis=0), np.concatenate(results_infrared, axis=0)]
    results = np.stack(dgps_arrays, axis=0)
    results = np.round(np.sqrt(results), 3)

    # Times:
    dgps_arrays_times = [np.concatenate(times_boston, axis=0), np.concatenate(times_communities, axis=0), np.concatenate(times_abalone, axis=0),
                   np.concatenate(times_ozone, axis=0), np.concatenate(times_forest, axis=0), np.concatenate(times_infrared, axis=0)]
    results_times = np.stack(dgps_arrays_times, axis=0)
    results_times = np.round(results_times, 3)

    # n_leaves:
    dgps_arrays_n_leaves = [np.concatenate(n_leaves_boston, axis=0), np.concatenate(n_leaves_communities, axis=0), np.concatenate(n_leaves_abalone, axis=0),
                     np.concatenate(n_leaves_ozone, axis=0), np.concatenate(n_leaves_forest, axis=0), np.concatenate(n_leaves_infrared, axis=0)]
    results_n_leaves = np.stack(dgps_arrays_n_leaves, axis=0)
    # Median RMSPE
    median = np.median(results, axis=1)
    deviation = np.mean(np.abs(results - median[:, np.newaxis, :]), axis=1)
    # Times:
    median_times = np.median(results_times, axis=1)
    deviation_times = np.mean(np.abs(results_times - median_times[:, np.newaxis, :]), axis=1)
    # n_leaves:
    median_n_leaves = np.median(results_n_leaves, axis=1)

    df = pd.DataFrame(median_n_leaves, index=['Boston', 'Communities', 'Abalone', 'Ozone', 'Forest', 'Infrared'],
                      columns=['Pruning', 'Global', 'Two-step', 'Semi'])
    print(df)

    column_names = ["Deep", "Local", "Pruning", "Two-Step", "Global", "Global Inter", "Semi"]

    #Specify the path:
    #dir = '/Users/ratmir/PycharmProjects/RandomForest'
    dir = '/home/RDC/miftachr/H:/miftachr/ES_CART'


    # Prints and saves:
    row_names = ['Boston', 'Communities', 'Abalone', 'Ozone', 'Forest', 'Infrared']
    print(simple_latex_table(median, deviation, column_names, row_names, name_threshold='median_empirical', table_dir=dir))
    print(simple_latex_table(median_times, deviation_times, column_names, row_names, name_threshold='median_empirical_times', table_dir=dir))

if __name__ == "__main__":
    main()