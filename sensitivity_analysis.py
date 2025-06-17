import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def load_data(filename):
    """Loads and returns the results_log from a pickle file."""
    if not os.path.exists(filename):
        print(f"Warning: File not found - {filename}")
        return None
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        if isinstance(data, dict) and 'results_log' in data:
            return data['results_log']
        elif isinstance(data, list):
            return data
    return None


def get_sensitivity_data(results_log, fixed_params, varying_param, k_value=20):
    """
    Extracts accuracy data for a sensitivity plot.

    Args:
        results_log (list): The list of all experiment results.
        fixed_params (dict): A dictionary of the parameters to hold constant.
        varying_param (str): The name of the parameter to vary.
        k_value (int): The k-value to use for accuracy.

    Returns:
        tuple: A tuple containing two lists (x_values, y_values) for plotting.
    """
    plot_data = {}

    for item in results_log:
        params = item.get('params', {})

        # Check if the fixed parameters match
        is_match = all(params.get(key) == val for key, val in fixed_params.items())

        if is_match:
            varying_value = params.get(varying_param)
            all_accs = item.get('all_accuracies', {})
            accuracy = all_accs.get(k_value, all_accs.get(str(k_value), -1.0))

            # We only care about valid runs
            if accuracy != -1.0:
                plot_data[varying_value] = accuracy

    # Sort the data by the varying parameter's value
    sorted_items = sorted(plot_data.items())
    x_values = [item[0] for item in sorted_items]
    y_values = [item[1] for item in sorted_items]

    return x_values, y_values


def run_sensitivity_analysis():
    """
    Main function to load all data and generate the 3-panel sensitivity plot.
    """
    # --- Configuration ---
    # Define the "best" parameters we found for each dataset based on k=20
    best_params = {
        'Mesh': {'gw_pi_sigma': 0.5, 'gw_pi_res': 10, 'gw_pi_weight_power': 1},
        'SHREC14': {'gw_pi_sigma': 0.05, 'gw_pi_res': 10, 'gw_pi_weight_power': 2},
        'ModelNet10': {'gw_pi_sigma': 2, 'gw_pi_res': 10, 'gw_pi_weight_power': 0.1}
    }

    # Map dataset names to their filenames
    filenames = {
        'Mesh': 'mesh_grid_search_results_20250616-182954.pkl',
        'SHREC14': 'shrec_grid_search_results_20250608-110157.pkl',
        'ModelNet10': 'modelnet10_gw_grid_search_results_20250610-073839.pkl'
    }

    # Define the desired order for plotting and legend
    plot_order = ['SHREC14', 'ModelNet10', 'Mesh']

    # Load data for all datasets
    all_data = {name: load_data(fname) for name, fname in filenames.items()}

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-paper')
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    # Academic, colorblind-friendly colors
    colors = {'Mesh': '#0072B2', 'SHREC14': '#009E73', 'ModelNet10': '#D55E00'}

    param_keys = ['gw_pi_sigma', 'gw_pi_res', 'gw_pi_weight_power']

    # Use formal names and LaTeX for labels
    x_labels = {
        'gw_pi_sigma': r'PI Bandwidth ($\sigma$)',
        'gw_pi_res': r'PI Resolution ($N$)',
        'gw_pi_weight_power': r'Weighting Power ($w_p$)'
    }

    for i, varying_param in enumerate(param_keys):
        ax = axes[i]

        # Loop in the specified order
        for dataset_name in plot_order:
            results_log = all_data.get(dataset_name)
            if results_log is None:
                continue

            current_best = best_params[dataset_name]
            fixed_params = {p: v for p, v in current_best.items() if p != varying_param}

            x_vals, y_vals = get_sensitivity_data(results_log, fixed_params, varying_param, k_value=20)

            y_vals_percent = [y * 100 for y in y_vals]

            if varying_param == 'gw_pi_res' and 0 in x_vals:
                zero_idx = x_vals.index(0)
                x_vals.pop(zero_idx)
                y_vals_percent.pop(zero_idx)

            ax.plot(x_vals, y_vals_percent, marker='o', markersize=10, linestyle='-', label=dataset_name,
                    color=colors[dataset_name], linewidth=3.5)

            best_val = current_best[varying_param]
            if best_val in x_vals:
                best_idx = x_vals.index(best_val)
                ax.plot(best_val, y_vals_percent[best_idx], '*', markersize=25, color=colors[dataset_name],
                        markeredgecolor='black', zorder=10)

        ax.set_xlabel(x_labels[varying_param], fontsize=20, labelpad=15)
        ax.xaxis.set_label_position('top')
        ax.tick_params(axis='x', labelbottom=True, labeltop=False, labelsize=16, pad=10)
        ax.tick_params(axis='y', which='major', labelsize=16)
        ax.grid(True, which='both', linestyle='--', linewidth=0.7)

        if varying_param == 'gw_pi_sigma':
            ax.set_xscale('log')

    axes[0].set_ylabel('Accuracy (%)', fontsize=20)

    handles, labels = axes[0].get_legend_handles_labels()
    # Create a new ordered list of handles and labels for the legend
    ordered_legend_elements = {label: handle for handle, label in zip(handles, labels)}
    final_handles = [ordered_legend_elements[label] for label in plot_order if label in ordered_legend_elements]
    final_labels = [label for label in plot_order if label in ordered_legend_elements]

    # Use subplots_adjust to create space, then place the legend
    fig.subplots_adjust(bottom=0.2, top=0.8)  # Create space at the bottom and top
    fig.legend(final_handles, final_labels, loc='lower center', ncol=3, fontsize=22)

    output_filename = "sensitivity_analysis.pdf"
    plt.savefig(output_filename, dpi=1600, bbox_inches='tight')
    print(f"\nAnalysis complete. Plot saved as '{output_filename}'")


if __name__ == "__main__":
    run_sensitivity_analysis()
