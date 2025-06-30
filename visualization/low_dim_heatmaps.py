# Import required libraries and modules
import matplotlib.pyplot as plt
import numpy as np
import importlib
import os
import sys

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Reload modules to ensure changes are applied
import data_generation
importlib.reload(data_generation)

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



f_rect, nuisance = data_generation.generate_data_from_X(X_train, noise_level=noise_level,
                                                   dgp_name='rectangular',
                                                   n_points=n_points,
                                                   add_noise=False)
f_circ, nuisance = data_generation.generate_data_from_X(X_train, noise_level=noise_level,
                                                   dgp_name='circular',
                                                   n_points=n_points,
                                                   add_noise=False)
f_sine, nuisance = data_generation.generate_data_from_X(X_train, noise_level=noise_level,
                                                   dgp_name='sine_cosine',
                                                   n_points=n_points,
                                                   add_noise=False)
f_smooth, nuisance = data_generation.generate_data_from_X(X_train, noise_level=noise_level,
                                                   dgp_name='smooth_signal',
                                                   n_points=n_points,
                                                   add_noise=False)



def plot_heatmap(data, name, cmap='viridis', save_path=None):

    fig, ax = plt.subplots(figsize=(10, 6))  # Create figure and axis

    # Create heatmap
    im = ax.imshow(data, origin='lower', extent=[0, 1, 0, 1], aspect='equal', cmap=cmap)

    # Add color bar with manual positioning
    cbar_ax = fig.add_axes([0.78, 0.08, 0.035, 0.88])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)

    # Manually set exactly six ticks for the colorbar, rounded to the nearest integer
    min_val, max_val = im.get_clim()  # Get the data limits from the heatmap
    ticks = np.linspace(min_val, max_val, 6)  # Generate 6 equally spaced tick values
    rounded_ticks = np.round(ticks).astype(int)  # Round to the nearest integer

    cbar.set_ticks(rounded_ticks)  # Set the rounded ticks on the colorbar
    cbar.set_ticklabels(rounded_ticks)  # Use the rounded integers as tick labels

    cbar.ax.tick_params(labelsize=18)  # Adjust color bar label size

    # Set major ticks only at 0 and 1 for both x and y axes
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xticklabels(['0', '', '', '', '', '1'], fontsize=18)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels(['0', '', '', '', '', '1'], fontsize=18)
    ax.tick_params(axis='both', which='major', length=5, width=2)  # Adjust tick mark length and width

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(os.path.join(save_path, f'heatmap_{name}.png'), dpi=300, bbox_inches='tight')

    # Display the plot
    plt.show()


func_reshaped_true_rect = f_rect.reshape(int(n_points/2), int(n_points/2))
func_reshaped_true_circ = f_circ.reshape(int(n_points/2), int(n_points/2))
func_reshaped_true_sine = f_sine.reshape(int(n_points/2), int(n_points/2))
func_reshaped_true_smooth = f_smooth.reshape(int(n_points/2), int(n_points/2))

plot_heatmap(data = func_reshaped_true_rect, save_path='/Users/ratmir/PycharmProjects/RandomForest', name='rect')
plot_heatmap(data = func_reshaped_true_circ, save_path='/Users/ratmir/PycharmProjects/RandomForest', name='circ')
plot_heatmap(data = func_reshaped_true_smooth, save_path='/Users/ratmir/PycharmProjects/RandomForest', name='smooth')
plot_heatmap(data = func_reshaped_true_sine, save_path='/Users/ratmir/PycharmProjects/RandomForest', name='sine')






