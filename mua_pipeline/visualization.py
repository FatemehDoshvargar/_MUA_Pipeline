# mua_pipeline/visualization.py
"""
Visualization utilities for MUA Pipeline
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import pearsonr


# Plot the results
def plot_results(predictions, actual, title=None):
    """
    Plot prediction results with scatter plot and error distribution.
    """
    if predictions is None:
        return

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.scatter(actual, predictions, alpha=0.7, s=30, edgecolors='k',
                linewidth=0.5, color='#4472C4', zorder=3)

    min_val_x = actual.min()
    max_val_x = actual.max()

    # Calculate correlation and statistics
    r_val, p_val = pearsonr(actual, predictions)

    # Calculate regression line from correlation coefficient
    mean_actual = np.mean(actual)
    mean_pred = np.mean(predictions)
    std_actual = np.std(actual)
    std_pred = np.std(predictions)

    # Regression line slope and intercept from correlation
    slope = r_val * (std_pred / std_actual)
    intercept = mean_pred - slope * mean_actual

    # Plot the regression line across the actual data range
    x_line = np.array([min_val_x, max_val_x])
    y_line = slope * x_line + intercept
    ax1.plot(x_line, y_line, 'r-', lw=2, alpha=0.8, zorder=2)

    ax1.set_xlabel('Actual Values', fontsize=11, fontweight='normal')
    ax1.set_ylabel('Predicted Values', fontsize=11, fontweight='normal')

    # Format p-value for display
    if p_val < 0.001:
        p_text = "p < 0.001"
    else:
        p_text = f"p = {p_val:.3f}"

    # Add text (removed R²)
    ax1.text(0.05, 0.95, f'r = {r_val:.3f}\n{p_text}',
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax1.grid(True, alpha=0.3, linewidth=0.5)
    ax1.set_axisbelow(True)

    # Error distribution
    errors = predictions - actual
    n, bins, patches = ax2.hist(errors, bins=20, edgecolor='k', linewidth=0.5,
                                alpha=0.8, color='#70AD47', density=True)
    ax2.axvline(0, color='k', linestyle='--', linewidth=1.5, alpha=0.8)

    mu, std = stats.norm.fit(errors)
    xmin, xmax = ax2.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    ax2.plot(x, p, 'k-', linewidth=1.5, alpha=0.8)

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    ax2.set_xlabel('Prediction Error', fontsize=11, fontweight='normal')
    ax2.set_ylabel('Density', fontsize=11, fontweight='normal')

    ax2.text(0.95, 0.95, f'MAE = {mae:.3f}\nRMSE = {rmse:.3f}',
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Grid for y-axis only
    ax2.grid(True, alpha=0.3, linewidth=0.5, axis='y')
    ax2.set_axisbelow(True)

    # Remove top and right spines for cleaner look
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='out', length=4, width=0.8)

    # Add panel labels
    ax1.text(-0.1, 1.05, 'A',
             transform=ax1.transAxes, fontsize=12, fontweight='bold')
    ax2.text(-0.1, 1.05, 'B',
             transform=ax2.transAxes, fontsize=12, fontweight='bold')

    plt.tight_layout()

    if title:
        fig.suptitle(title, y=1.02)

    plt.show()
