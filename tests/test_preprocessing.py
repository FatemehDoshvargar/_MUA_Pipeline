# tests/test_preprocessing.py
"""
Tests for preprocessing functions
"""

import numpy as np
import pytest

from mua_pipeline import preprocess


class TestPreprocess:
    """Tests for preprocess function"""

    def test_no_missing_data(self):
        """Test when there's no missing data"""
        connectivity = np.random.randn(10, 5, 5)
        behavior = np.random.randn(10)

        clean_conn, clean_behav, removed = preprocess(
            connectivity, behavior, missing_strategy='any', verbose=False
        )

        assert clean_conn.shape == connectivity.shape
        assert clean_behav.shape == behavior.shape
        assert len(removed) == 0

    def test_remove_zero_behavioral(self):
        """Test removing subjects with zero behavioral scores"""
        connectivity = np.random.randn(10, 5, 5)
        behavior = np.random.randn(10)
        behavior[2] = 0
        behavior[5] = 0

        clean_conn, clean_behav, removed = preprocess(
            connectivity, behavior, missing_strategy='zero', verbose=False
        )

        assert clean_conn.shape == (8, 5, 5)
        assert clean_behav.shape == (8,)
        assert len(removed) == 2
        assert 2 in removed
        assert 5 in removed

    def test_remove_nan_behavioral(self):
        """Test removing subjects with NaN behavioral scores"""
        connectivity = np.random.randn(10, 5, 5)
        behavior = np.random.randn(10)
        behavior[1] = np.nan
        behavior[7] = np.nan

        clean_conn, clean_behav, removed = preprocess(
            connectivity, behavior, missing_strategy='nan', verbose=False
        )

        assert clean_conn.shape == (8, 5, 5)
        assert len(removed) == 2

    def test_remove_inf_behavioral(self):
        """Test removing subjects with Inf behavioral scores"""
        connectivity = np.random.randn(10, 5, 5)
        behavior = np.random.randn(10)
        behavior[3] = np.inf
        behavior[8] = -np.inf

        clean_conn, clean_behav, removed = preprocess(
            connectivity, behavior, missing_strategy='inf', verbose=False
        )

        assert clean_conn.shape == (8, 5, 5)
        assert len(removed) == 2

    def test_remove_any_missing(self):
        """Test removing subjects with any type of missing data"""
        connectivity = np.random.randn(10, 5, 5)
        behavior = np.random.randn(10)
        behavior[0] = 0
        behavior[3] = np.nan
        behavior[7] = np.inf

        clean_conn, clean_behav, removed = preprocess(
            connectivity, behavior, missing_strategy='any', verbose=False
        )

        assert clean_conn.shape == (7, 5, 5)
        assert len(removed) == 3

    def test_2d_connectivity_input(self):
        """Test with 2D connectivity input"""
        connectivity = np.random.randn(10, 50)  # Already vectorized
        behavior = np.random.randn(10)
        behavior[2] = 0

        clean_conn, clean_behav, removed = preprocess(
            connectivity, behavior, missing_strategy='zero', verbose=False
        )

        assert clean_conn.shape == (9, 50)
        assert len(removed) == 1

    def test_transpose_detection_2d(self):
        """Test auto-detection and transposing of 2D data"""
        # Features x subjects (wrong orientation)
        connectivity = np.random.randn(50, 10)
        behavior = np.random.randn(10)

        clean_conn, clean_behav, removed = preprocess(
            connectivity, behavior, missing_strategy='any', verbose=False
        )

        # Should be transposed to subjects x features
        assert clean_conn.shape[0] == 10
        assert clean_conn.shape[1] == 50

    def test_error_on_mismatched_subjects(self):
        """Test error when subject counts don't match"""
        connectivity = np.random.randn(10, 5, 5)
        behavior = np.random.randn(8)  # Wrong number of subjects

        with pytest.raises(ValueError, match="Subject count mismatch"):
            preprocess(
                connectivity, behavior, verbose=False
            )
