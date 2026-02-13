# mua_pipeline/core.py
"""
Core classes for MUA Pipeline: FeatureVectorizer and MUA
Author: Fatemeh Doshvargar
"""

import numpy as np
from scipy.stats import rankdata
from scipy import stats  
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler  


# feature vectorizer
class FeatureVectorizer(BaseEstimator, TransformerMixin):
    """
    Transform connectivity matrices to feature vectors.

    Can handle both:
    - 3D input: (n_subjects, n_regions, n_regions) -
        extracts upper triangular elements
    - 2D input: (n_subjects, n_features) - passes through
        unchanged

    Parameters
    ----------
    verbose : bool, default=False
        Whether to print information during transformation

    Attributes
    ----------
    input_type_ : str
        Either '2D' or '3D' based on input data
    n_regions_ : int
        Number of regions in the connectivity matrix
            (only for 3D input)
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
            Either 2D (n_subjects, n_features) or 3D
                (n_subjects, n_regions, n_regions)
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
                print("FeatureVectorizer fitted (2D passthrough mode):")
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
                print("FeatureVectorizer fitted (3D vectorization mode):")
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
                raise ValueError(
                    "Expected matrices of shape ",
                    "({self.n_regions_}, {self.n_regions_}), "
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
    filter_by_sign : bool, default=False
        Main control parameter:
        - True: Split features into positive and negative networks 
        - False: Keep all features together 

    direction : str, default='difference'
        Only used when filter_by_sign=True. Controls how network scores are formed:
        - 'difference': Single score = mean(pos_edges) - mean(neg_edges) (original MATLAB CPM)
        - 'positive': Single column with positive network score only
        - 'negative': Single column with negative network score only
        Ignored when filter_by_sign=False.

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

    feature_aggregation : str, default='mean'
        How to aggregate features:
        - 'sum': Sum of features
        - 'mean': Mean of features (original CPM default, scale-invariant)

    standardize_scores : bool, default=False
        Whether to standardize the final aggregated scores (z-score normalization).
        - False: Keep raw scores
        - True: Standardize to mean=0, std=1
    """

    def __init__(self, filter_by_sign=False, direction='difference',
                 selection_method='pvalue', selection_threshold=0.05,
                 weighting_method='binary', correlation_type='pearson',
                 feature_aggregation='mean', standardize_scores=False):

        self.filter_by_sign = filter_by_sign
        self.direction = direction
        self.selection_method = selection_method
        self.selection_threshold = selection_threshold
        self.weighting_method = weighting_method
        self.correlation_type = correlation_type
        self.feature_aggregation = feature_aggregation
        self.standardize_scores = standardize_scores

    def fit(self, X, y):
        n_samples, n_edges = X.shape

        # Validate direction
        if self.filter_by_sign and self.direction not in ('difference', 'positive', 'negative'):
            raise ValueError(f"direction must be 'difference', 'positive', or 'negative', "
                             f"got '{self.direction}'")

        # Compute correlations
        self.correlations_, self.p_values_ = self._compute_correlations(X, y)

        # Select edges
        self.selected_edges_ = self._select_edges(n_edges)
        self.n_selected_edges_ = np.sum(self.selected_edges_)

        # Calculate edge weights
        self.edge_weights_ = self._calculate_edge_weights(X, y)

        # Store network masks if filter by sign
        if self.filter_by_sign:
            self.pos_mask_ = (self.edge_weights_ > 0) & self.selected_edges_
            self.neg_mask_ = (self.edge_weights_ < 0) & self.selected_edges_
            self.n_positive_ = np.sum(self.pos_mask_)
            self.n_negative_ = np.sum(self.neg_mask_)

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
        if self.filter_by_sign:
            return self._create_split_scores(X)
        else:
            return self._create_combined_scores(X)

    def _create_split_scores(self, X):
        """
        Create network scores for filter-by-sign mode.

        Depending on direction:
        - 'difference':  returns (n_samples, 1) with mean(pos_edges) - mean(neg_edges)
                         (matches original MATLAB cpm_train)
        - 'positive':    returns (n_samples, 1) with positive network score only
        - 'negative':    returns (n_samples, 1) with negative network score only
        """
        n_samples = X.shape[0]

        pos_mask = self.pos_mask_
        neg_mask = self.neg_mask_

        # Compute positive network score
        if np.any(pos_mask):
            pos_weights = np.abs(self.edge_weights_[pos_mask])
            weighted_pos = X[:, pos_mask] * pos_weights
            if self.feature_aggregation == 'sum':
                pos_score = np.sum(weighted_pos, axis=1)
            else:  # mean
                pos_score = np.mean(weighted_pos, axis=1)
        else:
            pos_score = np.zeros(n_samples)

        # Compute negative network score
        if np.any(neg_mask):
            neg_weights = np.abs(self.edge_weights_[neg_mask])
            weighted_neg = X[:, neg_mask] * neg_weights
            if self.feature_aggregation == 'sum':
                neg_score = np.sum(weighted_neg, axis=1)
            else:  # mean
                neg_score = np.mean(weighted_neg, axis=1)
        else:
            neg_score = np.zeros(n_samples)

        # Return based on direction
        if self.direction == 'difference':
            scores = (pos_score - neg_score).reshape(-1, 1)
        elif self.direction == 'positive':
            scores = pos_score.reshape(-1, 1)
        elif self.direction == 'negative':
            scores = neg_score.reshape(-1, 1)

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
