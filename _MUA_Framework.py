"""
Mass univariate aggregation methods for machine learning in neuroscience
Author: Fatemeh Doshvargar
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, rankdata
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import time
import HCP

# Preprocessing

def remove_subjects_with_missing_data(connectivity_matrix, behavioral_data, missing_strategy='any', verbose=True):
    """
    Remove subjects with missing data from connectivity and behavioral data.

    This function handles various input formats and automatically detects the
    correct orientation of the data, converting to standard format
    (subjects × features for 2D, subjects × regions × regions for 3D).

    Parameters
    ----------
    connectivity_matrix : array-like
        Either:
        - 2D: subjects × features (true format) or features × subjects
        - 3D: subjects × regions × regions (true format) or other orientations

    behavioral_data : array-like
        Either:
        - 1D: Behavioral scores (subjects,)
        - 2D: Behavioral scores (subjects, features) or (features, subjects)

    missing_strategy : str, default='any'
        Strategy for identifying missing data:
        - 'zero': behavioral_data == 0
        - 'nan': NaN values in behavioral_data
        - 'inf': inf/-inf values in behavioral_data
        - 'any': zero, NaN, or inf values in behavioral_data

    verbose : bool, default=True
        Whether to print information about removed subjects

    Returns
    -------
    clean_connectivity : array-like
        Connectivity data with missing subjects removed
    clean_behavioral : array-like
        Behavioral data with missing subjects removed
    removed_indices : array-like
        Indices of subjects that were removed
    """

    behavioral_data = np.array(behavioral_data)
    connectivity_matrix = np.array(connectivity_matrix)
    original_connectivity_shape = connectivity_matrix.shape
    original_behavioral_shape = behavioral_data.shape

    # Convert behavioral_data to standard format (subjects, features)
    if behavioral_data.ndim == 1:
        behavioral_true_format = behavioral_data.reshape(-1, 1)
        n_subjects_behavioral = len(behavioral_data)
    elif behavioral_data.ndim == 2:
        if behavioral_data.shape[0] >= behavioral_data.shape[1]:
            behavioral_true_format = behavioral_data
            n_subjects_behavioral = behavioral_data.shape[0]
        else:
            behavioral_true_format = behavioral_data.T
            n_subjects_behavioral = behavioral_data.shape[1]
            if verbose:
                print(f"Behavioral data transposed from {behavioral_data.shape} to {behavioral_true_format.shape}")
    else:
        raise ValueError(f"Behavioral data must be 1D or 2D, got {behavioral_data.ndim}D")

    # Convert connectivity_matrix to standard format
    if connectivity_matrix.ndim == 2:
        # Auto-detect format based on behavioral data dimension
        if connectivity_matrix.shape[0] == n_subjects_behavioral:
            connectivity_true_format = connectivity_matrix
            n_subjects_connectivity = connectivity_matrix.shape[0]
        elif connectivity_matrix.shape[1] == n_subjects_behavioral:
            connectivity_true_format = connectivity_matrix.T
            n_subjects_connectivity = connectivity_matrix.shape[1]
            if verbose:
                print(f"Connectivity matrix transposed from {connectivity_matrix.shape} to {connectivity_true_format.shape}")
        else:
            if connectivity_matrix.shape[0] >= connectivity_matrix.shape[1]:
                connectivity_true_format = connectivity_matrix
                n_subjects_connectivity = connectivity_matrix.shape[0]
            else:
                connectivity_true_format = connectivity_matrix.T
                n_subjects_connectivity = connectivity_matrix.shape[1]

    elif connectivity_matrix.ndim == 3:
        # Auto-detect 3D format and convert to (subjects, regions, regions)
        shape = connectivity_matrix.shape

        if shape[0] == n_subjects_behavioral:
            connectivity_true_format = connectivity_matrix
            n_subjects_connectivity = shape[0]
        elif shape[1] == n_subjects_behavioral:
            connectivity_true_format = np.transpose(connectivity_matrix, (1, 0, 2))
            n_subjects_connectivity = shape[1]
        elif shape[2] == n_subjects_behavioral:
            connectivity_true_format = np.transpose(connectivity_matrix, (2, 0, 1))
            n_subjects_connectivity = shape[2]
        else:
            # Fallback heuristics for ambiguous cases
            if shape[1] == shape[2] and shape[0] != shape[1]:
                connectivity_true_format = connectivity_matrix
                n_subjects_connectivity = shape[0]
            elif shape[0] == shape[2] and shape[0] != shape[1]:
                connectivity_true_format = np.transpose(connectivity_matrix, (1, 0, 2))
                n_subjects_connectivity = shape[1]
            elif shape[0] == shape[1] and shape[0] != shape[2]:
                connectivity_true_format = np.transpose(connectivity_matrix, (2, 0, 1))
                n_subjects_connectivity = shape[2]
            else:
                connectivity_true_format = connectivity_matrix
                n_subjects_connectivity = shape[0]

    else:
        raise ValueError(f"Connectivity matrix must be 2D or 3D, got {connectivity_matrix.ndim}D")

    # Verify subject counts match
    if n_subjects_connectivity != n_subjects_behavioral:
        raise ValueError(f"Subject count mismatch: connectivity has {n_subjects_connectivity} subjects, "
                        f"behavioral has {n_subjects_behavioral} subjects")

    # Find subjects to remove based on strategy
    if missing_strategy == 'zero':
        missing_mask = behavioral_true_format == 0
    elif missing_strategy == 'nan':
        missing_mask = np.isnan(behavioral_true_format)
    elif missing_strategy == 'inf':
        missing_mask = np.isinf(behavioral_true_format)
    elif missing_strategy == 'any':
        missing_mask = (behavioral_true_format == 0) | np.isnan(behavioral_true_format) | np.isinf(behavioral_true_format)
    else:
        raise ValueError(f"Unknown missing_strategy: {missing_strategy}")

    # Get indices of subjects to remove
    subjects_with_missing = np.any(missing_mask, axis=1) if behavioral_true_format.ndim == 2 else missing_mask.flatten()
    removed_indices = np.where(subjects_with_missing)[0]

    if len(removed_indices) == 0:
        if verbose:
            print("No subjects with missing data found.")
        return connectivity_true_format, behavioral_true_format.squeeze(), removed_indices

    # Remove subjects from both datasets
    clean_connectivity = np.delete(connectivity_true_format, removed_indices, axis=0)
    clean_behavioral = np.delete(behavioral_true_format, removed_indices, axis=0)

    # Squeeze behavioral data if it was originally 1D
    if original_behavioral_shape == (n_subjects_behavioral,):
        clean_behavioral = clean_behavioral.squeeze()

    if verbose:
        print(f"Missing data removal ({missing_strategy} strategy):")
        print(f"  Original subjects: {n_subjects_behavioral}")
        print(f"  Removed subjects: {len(removed_indices)}")
        print(f"  Final subjects: {len(clean_behavioral) if clean_behavioral.ndim == 1 else clean_behavioral.shape[0]}")
        print(f"  Connectivity shape: {original_connectivity_shape} → {clean_connectivity.shape}")
        print(f"  Behavioral shape: {original_behavioral_shape} → {clean_behavioral.shape}")

    return clean_connectivity, clean_behavioral, removed_indices


def vectorize_3d(connectome_3d, verbose=True):
    """
    Convert 3D connectome to 2D feature matrix by extracting upper triangle.

    Parameters
    ----------
    connectome_3d : array-like of shape (n_subjects, n_regions, n_regions)
        3D connectivity matrices
    verbose : bool, default=True
        Print conversion information

    Returns
    -------
    X : array-like of shape (n_subjects, n_features)
        2D feature matrix where features are upper triangular elements
    upper_tri_indices : tuple
        Indices of upper triangular elements
    """
    connectome_3d = np.array(connectome_3d)

    if connectome_3d.ndim != 3:
        raise ValueError(f"Input must be 3D array, got {connectome_3d.ndim}D")

    n_subjects, n_regions_1, n_regions_2 = connectome_3d.shape

    if n_regions_1 != n_regions_2:
        raise ValueError(f"Connectivity matrices must be square")

    n_regions = n_regions_1
    upper_tri_indices = np.triu_indices(n_regions, k=1)
    n_features = len(upper_tri_indices[0])

    X = np.zeros((n_subjects, n_features))
    for subject in range(n_subjects):
        X[subject, :] = connectome_3d[subject][upper_tri_indices]

    if verbose:
        print(f"3D to 2D vectorization:")
        print(f"  Input shape: {connectome_3d.shape}")
        print(f"  Output shape: {X.shape}")
        print(f"  Features per subject: {n_features}")

    return X, upper_tri_indices


# feature vectorizer

class FeatureVectorizer(BaseEstimator, TransformerMixin):
    """
    Transform connectivity matrices to feature vectors.

    Can handle both:
    - 3D input: (n_subjects, n_regions, n_regions) - extracts upper triangular elements
    - 2D input: (n_subjects, n_features) - passes through unchanged

    Parameters
    ----------
    verbose : bool, default=False
        Whether to print information during transformation

    Attributes
    ----------
    input_type_ : str
        Either '2D' or '3D' based on input data
    n_regions_ : int
        Number of regions in the connectivity matrix (only for 3D input)
    n_features_ : int
        Number of features
    upper_tri_indices_ : tuple
        Indices of upper triangular elements (only for 3D input)
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

    def fit(self, X, y=None):
        """
        Fit the vectorizer to determine input type and dimensions.

        Parameters
        ----------
        X : array-like
            Either 2D (n_subjects, n_features) or 3D (n_subjects, n_regions, n_regions)
        y : ignored
            Not used, present for sklearn compatibility

        Returns
        -------
        self : object
            Fitted vectorizer
        """
        X = np.array(X)

        if X.ndim == 2:
            # Already vectorized data
            self.input_type_ = '2D'
            self.n_features_ = X.shape[1]

            if self.verbose:
                print(f"FeatureVectorizer fitted (2D passthrough mode):")
                print(f"  Input shape: {X.shape}")
                print(f"  Output shape: {X.shape}")

        elif X.ndim == 3:
            # 3D connectivity matrices
            self.input_type_ = '3D'
            n_subjects, n_regions_1, n_regions_2 = X.shape

            if n_regions_1 != n_regions_2:
                raise ValueError("Connectivity matrices must be square")

            self.n_regions_ = n_regions_1
            self.upper_tri_indices_ = np.triu_indices(self.n_regions_, k=1)
            self.n_features_ = len(self.upper_tri_indices_[0])

            if self.verbose:
                print(f"FeatureVectorizer fitted (3D vectorization mode):")
                print(f"  Input shape: {X.shape}")
                print(f"  Output shape: ({n_subjects}, {self.n_features_})")

        else:
            raise ValueError(f"Input must be 2D or 3D array, got {X.ndim}D")

        return self

    def transform(self, X):
        """
        Transform input data to 2D feature matrix.

        Parameters
        ----------
        X : array-like
            Either 2D or 3D array, must match the type seen in fit()

        Returns
        -------
        X_transformed : array-like of shape (n_subjects, n_features)
            2D feature matrix
        """
        if not hasattr(self, 'input_type_'):
            raise ValueError("Vectorizer not fitted yet. Call fit() first.")

        X = np.array(X)

        if self.input_type_ == '2D':
            # Passthrough for 2D data
            if X.ndim != 2:
                raise ValueError(f"Expected 2D input, got {X.ndim}D")
            return X

        else:  # 3D
            if X.ndim != 3:
                raise ValueError(f"Expected 3D input, got {X.ndim}D")

            n_subjects = X.shape[0]

            if X.shape[1] != self.n_regions_ or X.shape[2] != self.n_regions_:
                raise ValueError(f"Expected matrices of shape ({self.n_regions_}, {self.n_regions_}), "
                               f"got ({X.shape[1]}, {X.shape[2]})")

            # Extract upper triangular elements
            X_transformed = np.zeros((n_subjects, self.n_features_))
            for i in range(n_subjects):
                X_transformed[i, :] = X[i][self.upper_tri_indices_]

            return X_transformed

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X_transformed):
        """
        Convert 2D feature matrix back to original format.

        Parameters
        ----------
        X_transformed : array-like of shape (n_subjects, n_features)
            2D feature matrix

        Returns
        -------
        X : array-like
            Original format data (2D passthrough or 3D reconstructed matrices)
        """
        if not hasattr(self, 'input_type_'):
            raise ValueError("Vectorizer not fitted yet. Call fit() first.")

        X_transformed = np.array(X_transformed)

        if self.input_type_ == '2D':
            # Passthrough for 2D data
            return X_transformed

        else:  # 3D
            n_subjects = X_transformed.shape[0]

            # Reconstruct matrices
            X = np.zeros((n_subjects, self.n_regions_, self.n_regions_))

            for i in range(n_subjects):
                # Fill upper triangle
                X[i][self.upper_tri_indices_] = X_transformed[i, :]
                # Make symmetric
                X[i] = X[i] + X[i].T

            return X


class MUA(BaseEstimator, TransformerMixin):

    """
    Mass Univariate Aggregation (MUA) estimator for connectivity-based predictive modeling.

    Parameters
    ----------
    split_by_sign : bool, default=False
        Main control parameter:
        - True: Split features into positive and negative networks (CPM-style)
        - False: Keep all features together (combined score)

    selection_method : str, default='pvalue'
        How to select edges:
        - 'pvalue': Select features with p < threshold
        - 'top_k': Select top k features by absolute correlation
        - 'all': Use all edges

    selection_threshold : float, default=0.05
        Threshold for edge selection:
        - If selection_method='pvalue': p-value threshold
        - If selection_method='top_k': number of features to select
        - If selection_method='all': ignored

    weighting_method : str, default='binary'
        How to weight features:
        - 'binary': +1/-1 based on correlation sign only
        - 'correlation': Use correlation coefficients
        - 'squared_correlation': Use squared correlations (preserving sign)
        - 'regression': Beta weights from univariate regression

    correlation_type : str, default='pearson'
        Type of correlation: 'pearson' or 'spearman'

    use_final_regression : bool, default=True
        Whether to fit a regression model on aggregated features

    regression_type : str, optional
        Type of regression: 'linear regression', 'robust regression', 'ridge regression', 'lasso regression'

    feature_aggregation : str, default='sum'
        How to aggregate features:
        - 'sum': Sum of features (traditional CPM)
        - 'mean': Mean of features (normalized, scale-invariant)

    standardize_scores : bool, default=False
        Whether to standardize the final aggregated scores (z-score normalization).
        - False: Keep raw scores
        - True: Standardize to mean=0, std=1
    """


    def __init__(self, split_by_sign=False,
                 selection_method='pvalue', selection_threshold=0.05,
                 weighting_method='binary', correlation_type='pearson',
                 feature_aggregation='sum', standardize_scores=False):

        self.split_by_sign = split_by_sign
        self.selection_method = selection_method
        self.selection_threshold = selection_threshold
        self.weighting_method = weighting_method
        self.correlation_type = correlation_type
        self.feature_aggregation = feature_aggregation
        self.standardize_scores = standardize_scores

    def fit(self, X, y):
        n_samples, n_edges = X.shape

        # Compute correlations
        self.correlations_, self.p_values_ = self._compute_correlations(X, y)

        # Select edges
        self.selected_edges_ = self._select_edges(n_edges)
        self.n_selected_edges_ = np.sum(self.selected_edges_)

        # Calculate edge weights
        self.edge_weights_ = self._calculate_edge_weights(X, y)

        # Store network info if split by sign
        if self.split_by_sign:
            self.n_positive_ = np.sum((self.edge_weights_ > 0) & self.selected_edges_)
            self.n_negative_ = np.sum((self.edge_weights_ < 0) & self.selected_edges_)

        # Initialize score scaler if needed
        if self.standardize_scores:
            self.score_scaler_ = StandardScaler()
            scores = self._create_scores(X)
            self.score_scaler_.fit(scores)

        return self

    def transform(self, X):
        if not hasattr(self, 'edge_weights_'):
            raise ValueError("Transformer not fitted yet. Call fit() first.")

        scores = self._create_scores(X)

        if self.standardize_scores:
            scores = self.score_scaler_.transform(scores)

        return scores

    def _compute_correlations(self, X, y):
        n_samples, n_edges = X.shape

        if self.correlation_type == 'spearman':
            y = rankdata(y)
            X = np.apply_along_axis(rankdata, axis=0, arr=X)

        y_mean = np.mean(y)
        y_std = np.std(y, ddof=1)
        if y_std == 0:
            return np.zeros(n_edges), np.ones(n_edges)
        y_z = (y - y_mean) / y_std

        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0, ddof=1)

        correlations = np.zeros(n_edges)
        p_values = np.ones(n_edges)

        valid_edges = X_std > 1e-10

        if np.any(valid_edges):
            X_z = np.zeros_like(X)
            X_z[:, valid_edges] = (X[:, valid_edges] - X_mean[valid_edges]) / X_std[valid_edges]

            correlations[valid_edges] = np.dot(X_z[:, valid_edges].T, y_z) / (n_samples - 1)

            t_stats = correlations[valid_edges] * np.sqrt(
                (n_samples - 2) / (1 - correlations[valid_edges] ** 2 + 1e-10))
            p_values[valid_edges] = 2 * (1 - stats.t.cdf(np.abs(t_stats), n_samples - 2))

        return correlations, p_values

    def _select_edges(self, n_edges):
        if self.selection_method == 'pvalue':
            selected_edges = self.p_values_ < self.selection_threshold
        elif self.selection_method == 'top_k':
            k = int(min(self.selection_threshold, n_edges))
            top_k_indices = np.argpartition(np.abs(self.correlations_), -k)[-k:]
            selected_edges = np.zeros(n_edges, dtype=bool)
            selected_edges[top_k_indices] = True
        else:  # 'all'
            selected_edges = np.ones(n_edges, dtype=bool)

        return selected_edges

    def _calculate_edge_weights(self, X, y):
        n_edges = X.shape[1]
        edge_weights = np.zeros(n_edges)

        if self.weighting_method == 'binary':
            edge_weights[self.selected_edges_ & (self.correlations_ > 0)] = 1.0
            edge_weights[self.selected_edges_ & (self.correlations_ < 0)] = -1.0

        elif self.weighting_method == 'correlation':
            edge_weights[self.selected_edges_] = self.correlations_[self.selected_edges_]

        elif self.weighting_method == 'squared_correlation':
            edge_weights[self.selected_edges_] = (
                    np.sign(self.correlations_[self.selected_edges_]) *
                    self.correlations_[self.selected_edges_] ** 2
            )

        elif self.weighting_method == 'regression':
            selected_indices = np.where(self.selected_edges_)[0]

            for idx in selected_indices:
                brain_edge = X[:, idx]
                XtX = np.dot(brain_edge, brain_edge)
                Xty = np.dot(brain_edge, y)

                if XtX > 0:
                    beta = Xty / XtX
                    edge_weights[idx] = beta
                else:
                    edge_weights[idx] = 0.0

        return edge_weights

    def _create_scores(self, X):
        if self.split_by_sign:
            return self._create_split_scores(X)
        else:
            return self._create_combined_scores(X)

    def _create_split_scores(self, X):
        n_samples = X.shape[0]

        pos_mask = (self.edge_weights_ > 0) & self.selected_edges_
        neg_mask = (self.edge_weights_ < 0) & self.selected_edges_

        scores = np.zeros((n_samples, 2))

        # Positive network
        if np.any(pos_mask):
            pos_weights = np.abs(self.edge_weights_[pos_mask])
            if self.feature_aggregation == 'sum':
                scores[:, 0] = np.sum(X[:, pos_mask] * pos_weights, axis=1)
            else:  # mean
                scores[:, 0] = np.mean(X[:, pos_mask] * pos_weights, axis=1)

        # Negative network
        if np.any(neg_mask):
            neg_weights = np.abs(self.edge_weights_[neg_mask])
            if self.feature_aggregation == 'sum':
                scores[:, 1] = np.sum(X[:, neg_mask] * neg_weights, axis=1)
            else:  # mean
                scores[:, 1] = np.mean(X[:, neg_mask] * neg_weights, axis=1)

        return scores

    def _create_combined_scores(self, X):
        selected_indices = np.where(self.selected_edges_)[0]
        n_samples = X.shape[0]

        scores = np.zeros((n_samples, 1))

        if len(selected_indices) > 0:
            selected_weights = self.edge_weights_[selected_indices]
            selected_edges = X[:, selected_indices]
            weighted_edges = selected_edges * selected_weights

            if self.feature_aggregation == 'sum':
                scores[:, 0] = np.sum(weighted_edges, axis=1)
            else:  # mean
                scores[:, 0] = np.mean(weighted_edges, axis=1)

        return scores

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
    ax1.text(-0.1, 1.05, 'A', transform=ax1.transAxes, fontsize=12, fontweight='bold')
    ax2.text(-0.1, 1.05, 'B', transform=ax2.transAxes, fontsize=12, fontweight='bold')

    plt.tight_layout()

    if title:
        fig.suptitle(title, y=1.02)

    plt.show()

if __name__ == "__main__":

    # Load data
    mat_file_path = "H:/untitled/s_hcp_fc_noble_corr.mat"
    connectome_data = HCP.reconstruct_fc_matrix(mat_file_path)
    behavioral_data = HCP.extract_behavioral_data_1d(mat_file_path, 'test1')

    # Preprocessing
    connectome_clean, behavioral_clean, removed_indices = remove_subjects_with_missing_data(
        connectome_data, behavioral_data
    )


    print("\n1. CPM Pipeline")

    cpm_pipeline = Pipeline([
        ('vectorize', FeatureVectorizer()),
        ('mua', MUA(
            split_by_sign=True,
            selection_method='pvalue',
            selection_threshold=0.05,
            weighting_method='binary',
            feature_aggregation='sum',
        )),
        ('regressor', LinearRegression())  # Linear regression
    ])

    # Cross-validation
    cpm_scores = cross_val_score(cpm_pipeline, connectome_clean, behavioral_clean, cv=10)
    cpm_predictions = cross_val_predict(cpm_pipeline, connectome_clean, behavioral_clean, cv=10)

    print(f"CPM R² (10-fold CV): {cpm_scores.mean():.3f} ± {cpm_scores.std():.3f}")

    # Evaluation
    cpm_r, cpm_p = pearsonr(behavioral_clean, cpm_predictions)
    mae = mean_absolute_error(behavioral_clean, cpm_predictions)
    rmse = np.sqrt(mean_squared_error(behavioral_clean, cpm_predictions))
    r2 = r2_score(behavioral_clean, cpm_predictions)

    print(f"Correlation: r={cpm_r:.3f}, p={cpm_p:.2e}")
    print(f"R²: {r2:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")


    print("\n2. PNRS Pipeline")

    pnrs_pipeline = Pipeline([
        ('vectorize', FeatureVectorizer()),
        ('mua', MUA(
            split_by_sign=False,
            selection_method='all',
            weighting_method='regression',
            feature_aggregation='sum',
        ))
    ])


    pnrs_scores = pnrs_pipeline.fit_transform(connectome_clean, behavioral_clean)

    # Use scores directly as predictions
    pnrs_predictions = pnrs_scores.flatten()

    # Evaluation
    pnrs_r, pnrs_p = pearsonr(behavioral_clean, pnrs_predictions)
    mae = mean_absolute_error(behavioral_clean, pnrs_predictions)
    rmse = np.sqrt(mean_squared_error(behavioral_clean, pnrs_predictions))

    r2 = r2_score(behavioral_clean, pnrs_predictions)

    print(f"PNRS scores shape: {pnrs_scores.shape}")
    print(f"Correlation: r={pnrs_r:.3f}, p={pnrs_p:.2e}")
    print(f"R²: {r2:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")


    print("SUMMARY")

    print("\nMethod\t\tCorrelation\tMAE")
    print("-" * 40)
    print(f"CPM\t\tr={cpm_r:.3f}\t\t{mean_absolute_error(behavioral_clean, cpm_predictions):.3f}")
    print(f"PNRS\t\tr={pnrs_r:.3f}\t\t{mean_absolute_error(behavioral_clean, pnrs_predictions):.3f}")

    # Plot results
    plot_results(cpm_predictions, behavioral_clean, title="CPM")
    plot_results(pnrs_predictions, behavioral_clean, title="PNRS")
