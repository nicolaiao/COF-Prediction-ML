import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors

np.random.seed(42)

def bootstrap_sampling(X, y, threshold=0.17, n_samples=500, thresh_region='high'):
    """
    Perform bootstrap sampling from the minority region.

    Parameters:
    X : pandas DataFrame, shape (n_samples, n_features)
        Feature matrix.
    y : pandas Series, shape (n_samples,)
        Target values.
    threshold : float, optional (default=0.17)
        Threshold to define the minority region.
    n_samples : int, optional (default=500)
        Number of bootstrap samples to generate.
    thresh_region : str, optional (default='high')
        Region of the threshold ('high' for values above, 'low' for values below).

    Returns:
    X_resampled : pandas DataFrame, shape (n_samples_new, n_features)
        Resampled feature matrix.
    y_resampled : pandas Series, shape (n_samples_new,)
        Resampled target values.
    """
    # Define the minority region based on the threshold
    if thresh_region == 'high':
        minority_indices = y[y > threshold].index
    elif thresh_region == 'low':
        minority_indices = y[y < threshold].index
    else:
        raise ValueError("thresh_region must be 'high' or 'low'")

    # Perform bootstrap sampling
    bootstrap_indices = np.random.choice(minority_indices, size=n_samples, replace=True)
    X_bootstrap = X.loc[bootstrap_indices]
    y_bootstrap = y.loc[bootstrap_indices]
    bootstrap_indices_original = X.index[bootstrap_indices]

    # Combine original and bootstrap samples
    X_resampled = pd.concat([X, X_bootstrap])
    y_resampled = pd.concat([y, y_bootstrap])

    return X_resampled, y_resampled

def bootstrap_sampling_with_noise(X, y, threshold=0.17, n_samples=500, thresh_region='high', noise_std=0.001, group_id_col='Group ID'):
    """
    Perform bootstrap sampling with optional Gaussian noise added to the samples.

    Parameters:
    X : pandas DataFrame, shape (n_samples, n_features)
        Feature matrix.
    y : pandas Series, shape (n_samples,)
        Target values.
    threshold : float, optional (default=0.17)
        Threshold to define the minority region.
    n_samples : int, optional (default=500)
        Number of bootstrap samples to generate.
    thresh_region : str, optional (default='high')
        Region of the threshold ('high' for values above, 'low' for values below).
    noise_std : float, optional (default=0.01)
        Standard deviation of the Gaussian noise to be added to the samples.

    Returns:
    X_resampled : pandas DataFrame, shape (n_samples_new, n_features)
        Resampled feature matrix with noise.
    y_resampled : pandas Series, shape (n_samples_new,)
        Resampled target values.
    """
    # Define the minority region based on the threshold
    if thresh_region == 'high':
        minority_indices = y[y > threshold].index
    elif thresh_region == 'low':
        minority_indices = y[y < threshold].index
    else:
        raise ValueError("thresh_region must be 'high' or 'low'")

    # Perform bootstrap sampling
    bootstrap_indices = np.random.choice(minority_indices, size=n_samples, replace=True)
    X_bootstrap = X.loc[bootstrap_indices]
    y_bootstrap = y.loc[bootstrap_indices]
    group_ids = X_bootstrap[group_id_col]

    # Separate the 'Group ID' column from the features
    X_bootstrap_features = X_bootstrap.drop(columns=[group_id_col])

    # Add Gaussian noise to the bootstrap samples (excluding 'Group ID')
    X_bootstrap_noisy = X_bootstrap_features + np.random.normal(0, noise_std, X_bootstrap_features.shape)
    y_bootstrap_noisy = y_bootstrap + np.random.normal(0, noise_std, y_bootstrap.shape)

    # Combine the noisy features with the 'Group ID' column
    X_bootstrap_noisy[group_id_col] = group_ids
    # Combine original and noisy bootstrap samples
    X_resampled = pd.concat([X, pd.DataFrame(X_bootstrap_noisy, columns=X.columns)])
    y_resampled = pd.concat([y, pd.Series(y_bootstrap_noisy, name=y.name)])

    return X_resampled, y_resampled

def bootstrap_sampling_with_knn(X, y, threshold=0.25, n_samples=100, thresh_region='high', k=2, group_id_col='Group ID'):
    """
    Perform bootstrap sampling with k-Nearest Neighbors (kNN) to generate synthetic samples from the minority region.

    Parameters:
    X : pandas DataFrame, shape (n_samples, n_features)
        Feature matrix.
    y : pandas Series or DataFrame, shape (n_samples,)
        Target values.
    threshold : float, optional (default=0.25)
        Threshold to define the minority region.
    n_samples : int, optional (default=100)
        Number of bootstrap samples to generate.
    thresh_region : str, optional (default='high')
        Region of the threshold ('high' for values above, 'low' for values below).
    k : int, optional (default=2)
        Number of nearest neighbors to use for generating synthetic samples.
    group_id_col : str, optional (default='Group ID')
        Column name for the group ID.

    Returns:
    X_resampled : pandas DataFrame, shape (n_samples_new, n_features)
        Resampled feature matrix.
    y_resampled : pandas Series or DataFrame, shape (n_samples_new,)
        Resampled target values.
    """
    # Convert to numpy arrays
    X_numpy = X.drop(columns=[group_id_col]).values
    group_ids = X[group_id_col].values
    y_numpy = y.values.flatten()  # Ensure y is a 1D array

    # Define the minority region based on the threshold
    if thresh_region == 'high':
        minority_indices = np.where(y_numpy > threshold)[0]
    elif thresh_region == 'low':
        minority_indices = np.where(y_numpy < threshold)[0]
    else:
        raise ValueError("thresh_region must be 'high' or 'low'")

    # Perform bootstrap sampling
    bootstrap_indices = np.random.choice(minority_indices, size=n_samples, replace=True)
    X_bootstrap = X_numpy[bootstrap_indices]
    group_ids_bootstrap = group_ids[bootstrap_indices]
    y_bootstrap = y_numpy[bootstrap_indices]
    bootstrap_indices_original = X.index[bootstrap_indices]

    # Generate synthetic samples using kNN
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_bootstrap)
    synthetic_X = []
    synthetic_y = []
    synthetic_group_ids = []

    for idx in range(len(bootstrap_indices)):
        neighbors = nbrs.kneighbors([X_bootstrap[idx]], return_distance=False)[0]
        neighbor_idx = np.random.choice(neighbors[1:])  # exclude the point itself

        # Generate synthetic sample by interpolation
        diff = X_bootstrap[neighbor_idx] - X_bootstrap[idx]
        lambda_ = np.random.rand()
        synthetic_sample = X_bootstrap[idx] + lambda_ * diff
        synthetic_target = y_bootstrap[idx] + lambda_ * (y_bootstrap[neighbor_idx] - y_bootstrap[idx])

        if synthetic_target > threshold:
            synthetic_X.append(synthetic_sample)
            synthetic_y.append(synthetic_target)
            synthetic_group_ids.append(group_ids_bootstrap[idx])
        else:
            print("Synthetic sample not in minority region. Skipping...")

    synthetic_X = np.array(synthetic_X)
    synthetic_y = np.array(synthetic_y)
    synthetic_group_ids = np.array(synthetic_group_ids)

    # Create DataFrames for synthetic data
    synthetic_X_df = pd.DataFrame(synthetic_X, columns=X.drop(columns=[group_id_col]).columns, index=bootstrap_indices_original)
    synthetic_y_df = pd.Series(synthetic_y, name=y.name, index=bootstrap_indices_original)
    synthetic_group_ids_df = pd.Series(synthetic_group_ids, name=group_id_col, index=bootstrap_indices_original)

    # Combine original and synthetic samples
    X_resampled = pd.concat([X, synthetic_X_df])
    X_resampled[group_id_col] = pd.concat([X[group_id_col], synthetic_group_ids_df])
    y_resampled = pd.concat([y, synthetic_y_df])

    return X_resampled, y_resampled



def bootstrap_sampling_with_difficulty(X, y, threshold=0.25, n_samples=100, thresh_region='high'):
    """
    Perform bootstrap sampling using linear regression to determine difficulty of prediction
    and generate synthetic samples from the minority region.

    Parameters:
    X : pandas DataFrame, shape (n_samples, n_features)
        Feature matrix.
    y : pandas Series or DataFrame, shape (n_samples,)
        Target values.
    threshold : float, optional (default=0.25)
        Threshold to define the minority region.
    n_samples : int, optional (default=100)
        Number of bootstrap samples to generate.
    thresh_region : str, optional (default='high')
        Region of the threshold ('high' for values above, 'low' for values below).

    Returns:
    X_resampled : pandas DataFrame, shape (n_samples_new, n_features)
        Resampled feature matrix.
    y_resampled : pandas Series or DataFrame, shape (n_samples_new,)
        Resampled target values.
    """
    # Convert to numpy arrays
    X_numpy = X.values
    y_numpy = y.values.flatten()  # Ensure y is a 1D array

    # Define the minority region based on the threshold
    if thresh_region == 'high':
        minority_indices = np.where(y_numpy > threshold)[0]
    elif thresh_region == 'low':
        minority_indices = np.where(y_numpy < threshold)[0]
    else:
        raise ValueError("thresh_region must be 'high' or 'low'")

    X_minority = X_numpy[minority_indices]
    y_minority = y_numpy[minority_indices]

    # Train a linear regression model
    lr = LinearRegression()
    lr.fit(X_minority, y_minority)

    # Calculate residuals (difficulty of prediction)
    residuals = np.abs(y_minority - lr.predict(X_minority))

    # Perform bootstrap sampling based on residuals
    prob_distribution = residuals / np.sum(residuals)
    bootstrap_indices = np.random.choice(minority_indices, size=n_samples, replace=True, p=prob_distribution)
    X_bootstrap = X_numpy[bootstrap_indices]
    y_bootstrap = y_numpy[bootstrap_indices]
    bootstrap_indices_original = X.index[bootstrap_indices]

    # Create DataFrames for bootstrap data
    X_bootstrap_df = pd.DataFrame(X_bootstrap, columns=X.columns, index=bootstrap_indices_original)
    y_bootstrap_df = pd.Series(y_bootstrap, name=y.name, index=bootstrap_indices_original)

    # Combine original and bootstrap samples
    X_resampled = pd.concat([X, X_bootstrap_df])
    y_resampled = pd.concat([y, y_bootstrap_df])

    return X_resampled, y_resampled
# train_X_bs2, train_y_bs2 = bootstrap_sampling_with_knn_difficulty(train_df.drop('cof', axis=1).to_numpy(), train_df['cof'].to_numpy(), n_samples=50, threshold=0.09, groups=train_df.index, thresh_region='low')

def plot_resampling(old_y, new_y, title="", path=None):
    # plot y distribution
    plt.figure()
    sns.kdeplot(new_y, label = "Modified")
    sns.kdeplot(old_y, label = "Original")
    plt.legend()
    plt.title(title, fontsize='large', fontweight='bold')
    if path is not None:
        plt.savefig(path)
    plt.show()