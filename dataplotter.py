import warnings
warnings.simplefilter(action='ignore', category=Warning)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

def plot_residuals(y_true, y_pred, title='Residual Plot', path=None, showbins=10):
    """
    Plot residuals of a machine learning model using seaborn.

    Args:
    y_true (array-like): True target values.
    y_pred (array-like): Predictions from the model.

    Returns:
    matplotlib figure: A plot showing the residuals.
    """
    sns.set_theme(color_codes=True)
    # Convert to NumPy arrays to ensure compatibility
    y_true = np.array(y_true).flatten()  # Flatten the array to 1D
    y_pred = np.array(y_pred).flatten()  # Flatten the array to 1D

    # Calculate residuals
    residuals = y_true - y_pred

    # Create a residual plot
    sns.residplot(x=y_pred, y=residuals, lowess=True)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.title(title, fontsize='large', fontweight='bold')
    # Set the maximum number of ticks on the x-axis
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=showbins))
    if path:
        plt.savefig(path)
    plt.show()
    

def plot_predictions(y_true, y_pred, title='Predicted vs Experimental COF', path=None, target="COF"):
    """
    Plot predictions of a machine learning model using seaborn.

    Args:
    y_true (array-like): True target values, potentially as nested arrays.
    y_pred (array-like): Predictions from the model, potentially as nested arrays.

    Returns:
    matplotlib figure: A plot showing the predictions.
    """
    sns.set_theme(color_codes=True)

    # Convert to NumPy arrays to ensure compatibility
    y_true = np.array(y_true).flatten()  # Flatten the array to 1D
    y_pred = np.array(y_pred).flatten()  # Flatten the array to 1D

    # Create the regression plot
    ax = sns.regplot(x=y_true, y=y_pred, lowess=True, scatter_kws={'alpha':0.8})

    # Determine the range of data for the perfect prediction line
    combined = np.concatenate([y_true, y_pred])
    min_val = np.min(combined)
    max_val = np.max(combined)

    # Adding the line of perfect prediction
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

    # Set labels and title with styling
    ax.set_xlabel(f'Experimental {target}')
    ax.set_ylabel(f'Predicted {target}')
    ax.set_title(title, fontsize='large', fontweight='bold')

    # Optional: add a legend to the plot
    ax.legend()
    if path:
        plt.savefig(path)
    # Show the plot
    plt.show()


def plot_results(y_true, y_pred, title="Predicted and True Values", path=None):
    sns.set_theme(color_codes=True)
    # Sort the true values and corresponding predicted values
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    sorted_indices = np.argsort(y_true)
    y_true_sorted = np.array(y_true)[sorted_indices]
    y_pred_sorted = np.array(y_pred)[sorted_indices]
    
    sns.set_theme(color_codes=True)
    plt.plot(y_true_sorted, label='True Values', color='b')
    plt.plot(y_pred_sorted, label='Predicted Values', color='r', alpha=0.7)
    plt.title(title, fontsize='large', fontweight='bold')
    plt.xlabel('Samples')
    plt.ylabel('Values')
    plt.legend()
    if path:
        plt.savefig(path)
    plt.show()
    
from matplotlib.ticker import MaxNLocator

def plot_residual_dist(y_true, y_pred, title='Residual Distribution', path=None, showbins=10):
    # Calculate residuals
    residuals = y_true - y_pred
    sns.set_theme(color_codes=True)
    # Plot the distribution of residuals
    sns.histplot(residuals, kde=True, bins=30, legend=False, color='b')
    plt.title(title, fontsize='large', fontweight='bold')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.axvline(0, color='r', linestyle='--', label='Zero Residuals')
    
    # Set the maximum number of ticks on the x-axis
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=showbins))
    
    if path:
        plt.savefig(path)
    plt.show()

