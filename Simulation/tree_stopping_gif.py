# Import required libraries and modules
import matplotlib.pyplot as plt
import numpy as np
import importlib
import os
import regression_tree as rt
import semiglobal as semi

# Reload modules to ensure changes are applied
import data_generation
importlib.reload(data_generation)
importlib.reload(rt)

np.random.seed(42)

# Get X:
n_points = 128  # Number of points in each dimension

noise_level = 1
X1, X2 = np.linspace(0, n_points, n_points), np.linspace(0, n_points, n_points)
X1_train, X2_train = X1[::2], X2[::2]
X1_train, X2_train = np.meshgrid(X1_train, X2_train)

X_train = np.c_[X1_train.ravel(), X2_train.ravel()]

X1_test, X2_test = X1[1::2], X2[1::2]
X1_test, X2_test = np.meshgrid(X1_test, X2_test)
X_test = np.c_[X1_test.ravel(), X2_test.ravel()]


y_train, noise_train = data_generation.generate_data_from_X(X_train, noise_level=noise_level, dgp_name='rectangular', n_points=n_points,
                                                            add_noise=True)

y_test, noise_est = data_generation.generate_data_from_X(X_test, noise_level=noise_level, dgp_name='rectangular', n_points=n_points,
                                                            add_noise=True)

f, nuisance = data_generation.generate_data_from_X(X_train, noise_level=noise_level, dgp_name='rectangular', n_points=n_points,
                                                   add_noise=False)
f_test, nuisance = data_generation.generate_data_from_X(X_test, noise_level=noise_level, dgp_name='rectangular', n_points=n_points,
                                                   add_noise=False)

kappa = 1

regression_tree = rt.RegressionTree(design=X_train, response=y_train, min_samples_split=1)
regression_tree.iterate(max_depth=35)
early_stopping_iteration = regression_tree.get_discrepancy_stop(critical_value=kappa)


tree_semi = semi.DecisionTreeRegressor(X_train, y_train, max_iter=3000,
                                       min_samples_split=1, kappa=kappa)
tree_semi.iterate(max_depth=40)
early_stopping_iteration_semi = tree_semi.stopping_iteration
stop = early_stopping_iteration_semi

# Create a directory to save the images
os.makedirs('gif_global_frames_playing', exist_ok=True)
vmins_list = []
vmax_list = []
# Find minimum and maximum values for the colorbar
for i in range(0, stop+1):
    print(i)
    #Global:
    predict_global = regression_tree.predict(X_train, depth=i)
    # Semi-global:
    tree_semi = semi.DecisionTreeRegressor(X_train, y_train, max_iter=i,
                                           min_samples_split=1, kappa=kappa)
    tree_semi.iterate(max_depth=10)
    predict_semi = tree_semi.predict(X_train)

    all_values = np.concatenate([y_train, predict_global, predict_semi])
    vmin, vmax = all_values.min(), all_values.max()
    vmins_list.append(vmin)
    vmax_list.append(vmax)

# Define a list to store filenames
filenames = []
vmin = min(vmins_list)
vmax = max(vmax_list)
# vmin = -1.2
# vmax = 1.2
# Generate and save each frame, fir .gif purpose:
for i in range(0, stop+1):
    print(i)
    #Global:
    predict_global = regression_tree.predict(X_train, depth=i)
    # Semi-global:
    tree_semi = semi.DecisionTreeRegressor(X_train, y_train, max_iter=i,
                                           min_samples_split=1, kappa=kappa)
    tree_semi.iterate(max_depth=10)
    predict_semi = tree_semi.predict(X_train)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4.5))

    # Heatmap 0
    func_reshaped_true = y_train.reshape(int(n_points/2), int(n_points/2))
    im = axs[0].imshow(func_reshaped_true, origin='lower', extent=[0, 1, 0, 1], aspect='equal', cmap='seismic', vmin=vmin, vmax=vmax)
    axs[0].axis('off')  # Disable axis for this subplot

    # Heatmap 1
    func_reshaped = predict_global.reshape(int(n_points/2), int(n_points/2))
    im = axs[1].imshow(func_reshaped, origin='lower', extent=[0, 1, 0, 1], aspect='equal', cmap='seismic', vmin=vmin, vmax=vmax)
    axs[1].axis('off')  # Disable axis for this subplot
    # Calculate 2^i
    axs[1].text(0.5, -0.1, f'Global: $k={2 ** i}$', fontsize=10, ha='center', va='center', transform=axs[1].transAxes)

    # Heatmap 2: local or semi-global
    func_reshaped = predict_semi.reshape(int(n_points/2), int(n_points/2))
    im = axs[2].imshow(func_reshaped, origin='lower', extent=[0, 1, 0, 1], aspect='equal', cmap='seismic', vmin=vmin, vmax=vmax)
    axs[2].axis('off')  # Disable axis for this subplot
    axs[2].text(0.5, -0.1, f'Semi-global: $k={i + 1}$', fontsize=10, ha='center', va='center',
                transform=axs[2].transAxes)

    # Save the figure
    filename = f'gif_global_frames_playing/frame_{i:02d}.png'
    plt.savefig(filename)
    filenames.append(filename)
    plt.close(fig)

