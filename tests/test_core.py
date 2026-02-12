# tests/test_core.py
"""
Tests for FeatureVectorizer and MUA classes
"""

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from mua_pipeline import FeatureVectorizer, MUA


class TestFeatureVectorizer:
    """Tests for FeatureVectorizer class"""

    @pytest.fixture
    def sample_3d_data(self):
        """Create sample 3D connectivity matrices"""
        np.random.seed(42)
        n_subjects, n_regions = 10, 20
        data = np.random.randn(n_subjects, n_regions, n_regions)
        # Make symmetric
        for i in range(n_subjects):
            data[i] = (data[i] + data[i].T) / 2
            np.fill_diagonal(data[i], 1.0)
        return data

    @pytest.fixture
    def sample_2d_data(self):
        """Create sample 2D feature matrix"""
        np.random.seed(42)
        return np.random.randn(10, 190)  # 10 subjects, 190 features

    def test_fit_3d_data(self, sample_3d_data):
        """Test fitting with 3D data"""
        vectorizer = FeatureVectorizer()
        vectorizer.fit(sample_3d_data)

        assert vectorizer.input_type_ == '3D'
        assert vectorizer.n_regions_ == 20
        assert vectorizer.n_features_ == 190  # 20*19/2

    def test_fit_2d_data(self, sample_2d_data):
        """Test fitting with 2D data"""
        vectorizer = FeatureVectorizer()
        vectorizer.fit(sample_2d_data)

        assert vectorizer.input_type_ == '2D'
        assert vectorizer.n_features_ == 190

    def test_transform_3d_data(self, sample_3d_data):
        """Test transforming 3D data to 2D"""
        vectorizer = FeatureVectorizer()
        transformed = vectorizer.fit_transform(sample_3d_data)

        assert transformed.shape == (10, 190)
        assert transformed.ndim == 2

    def test_transform_2d_data(self, sample_2d_data):
        """Test 2D data passes through unchanged"""
        vectorizer = FeatureVectorizer()
        transformed = vectorizer.fit_transform(sample_2d_data)

        np.testing.assert_array_equal(transformed, sample_2d_data)

    def test_inverse_transform_3d(self, sample_3d_data):
        """Test inverse transform reconstructs matrices"""
        vectorizer = FeatureVectorizer()
        transformed = vectorizer.fit_transform(sample_3d_data)
        reconstructed = vectorizer.inverse_transform(transformed)

        assert reconstructed.shape == sample_3d_data.shape
        # Check symmetry
        for i in range(reconstructed.shape[0]):
            np.testing.assert_array_almost_equal(
                reconstructed[i], reconstructed[i].T
            )

    def test_error_on_non_square_matrices(self):
        """Test error raised for non-square matrices"""
        vectorizer = FeatureVectorizer()
        bad_data = np.random.randn(10, 20, 15)  # Not square

        with pytest.raises(ValueError, match="must be square"):
            vectorizer.fit(bad_data)

    def test_error_on_unfitted_transform(self, sample_3d_data):
        """Test error when transforming before fitting"""
        vectorizer = FeatureVectorizer()

        with pytest.raises(ValueError, match="not fitted"):
            vectorizer.transform(sample_3d_data)


class TestMUA:
    """Tests for MUA class"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for MUA"""
        np.random.seed(42)
        n_subjects = 50
        n_features = 100

        X = np.random.randn(n_subjects, n_features)
        # Create y with some relationship to X
        y = np.sum(X[:, :10], axis=1) + np.random.randn(n_subjects) * 0.5

        return X, y

    def test_fit_split_by_sign(self, sample_data):
        """Test MUA with split_by_sign=True"""
        X, y = sample_data
        mua = MUA(split_by_sign=True, selection_method='pvalue',
                  selection_threshold=0.05)
        mua.fit(X, y)

        assert hasattr(mua, 'correlations_')
        assert hasattr(mua, 'p_values_')
        assert hasattr(mua, 'selected_edges_')
        assert hasattr(mua, 'n_positive_')
        assert hasattr(mua, 'n_negative_')

    def test_fit_combined(self, sample_data):
        """Test MUA with split_by_sign=False"""
        X, y = sample_data
        mua = MUA(split_by_sign=False, selection_method='all')
        mua.fit(X, y)

        assert hasattr(mua, 'edge_weights_')
        assert not hasattr(mua, 'n_positive_')

    def test_transform_split_scores(self, sample_data):
        """Test transform produces correct shape for split scores"""
        X, y = sample_data
        mua = MUA(split_by_sign=True)
        scores = mua.fit_transform(X, y)

        assert scores.shape == (50, 2)  # 50 subjects, 2 scores (pos/neg)

    def test_transform_combined_scores(self, sample_data):
        """Test transform produces correct shape for combined scores"""
        X, y = sample_data
        mua = MUA(split_by_sign=False)
        scores = mua.fit_transform(X, y)

        assert scores.shape == (50, 1)  # 50 subjects, 1 score

    def test_selection_methods(self, sample_data):
        """Test different selection methods"""
        X, y = sample_data

        # p-value selection
        mua_pval = MUA(selection_method='pvalue', selection_threshold=0.05)
        mua_pval.fit(X, y)
        n_selected_pval = np.sum(mua_pval.selected_edges_)

        # top_k selection
        mua_topk = MUA(selection_method='top_k', selection_threshold=20)
        mua_topk.fit(X, y)
        n_selected_topk = np.sum(mua_topk.selected_edges_)

        # all selection
        mua_all = MUA(selection_method='all')
        mua_all.fit(X, y)
        n_selected_all = np.sum(mua_all.selected_edges_)

        assert n_selected_topk == 20
        assert n_selected_all == 100
        assert 0 < n_selected_pval < 100

    def test_weighting_methods(self, sample_data):
        """Test different weighting methods"""
        X, y = sample_data

        methods = ['binary', 'correlation',
                   'squared_correlation', 'regression']

        for method in methods:
            mua = MUA(weighting_method=method, selection_method='all')
            mua.fit(X, y)
            scores = mua.transform(X)

            assert scores.shape[0] == 50
            assert not np.all(scores == 0)

    def test_sklearn_pipeline_integration(self, sample_data):
        """Test MUA works in sklearn pipeline"""
        X, y = sample_data

        pipeline = Pipeline([
            ('mua', MUA(split_by_sign=True, selection_method='pvalue')),
            ('regressor', LinearRegression())
        ])

        pipeline.fit(X, y)
        predictions = pipeline.predict(X)

        assert predictions.shape == (50,)

    def test_error_on_unfitted_transform(self, sample_data):
        """Test error when transforming before fitting"""
        X, y = sample_data
        mua = MUA()

        with pytest.raises(ValueError, match="not fitted"):
            mua.transform(X)


class TestIntegration:
    """Integration tests for full pipeline"""

    def test_full_cpm_pipeline(self):
        """Test complete CPM pipeline"""
        np.random.seed(42)
        n_subjects, n_regions = 30, 20

        # Create 3D connectivity data
        connectivity = np.random.randn(n_subjects, n_regions, n_regions)
        for i in range(n_subjects):
            connectivity[i] = (connectivity[i] + connectivity[i].T) / 2
            np.fill_diagonal(connectivity[i], 1.0)

        behavior = np.random.randn(n_subjects)

        # Build pipeline
        pipeline = Pipeline([
            ('vectorize', FeatureVectorizer()),
            ('mua', MUA(split_by_sign=True, selection_method='pvalue')),
            ('regressor', LinearRegression())
        ])

        # Fit and predict
        pipeline.fit(connectivity, behavior)
        predictions = pipeline.predict(connectivity)

        assert predictions.shape == (n_subjects,)
        assert not np.all(np.isnan(predictions))

    def test_full_pnrs_pipeline(self):
        """Test complete PNRS pipeline"""
        np.random.seed(42)
        n_subjects, n_regions = 30, 20

        connectivity = np.random.randn(n_subjects, n_regions, n_regions)
        for i in range(n_subjects):
            connectivity[i] = (connectivity[i] + connectivity[i].T) / 2
            np.fill_diagonal(connectivity[i], 1.0)

        behavior = np.random.randn(n_subjects)

        pipeline = Pipeline([
            ('vectorize', FeatureVectorizer()),
            ('mua', MUA(split_by_sign=False, selection_method='all',
                        weighting_method='regression'))
        ])

        scores = pipeline.fit_transform(connectivity, behavior)

        assert scores.shape == (n_subjects, 1)
        assert not np.all(np.isnan(scores))
