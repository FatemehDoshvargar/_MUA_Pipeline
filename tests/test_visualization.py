# tests/test_visualization.py
"""
Tests for visualization functions
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
from mua_pipeline import plot_results

matplotlib.use('Agg')  # Use non-interactive backend for testing


class TestPlotResults:
    """Tests for plot_results function"""

    def test_plot_creates_figure(self):
        """Test that plot_results creates a figure"""
        np.random.seed(42)
        predictions = np.random.randn(50)
        actual = predictions + np.random.randn(50) * 0.5

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plot_results(predictions, actual, title="Test")

        # Check that a figure was created
        assert len(plt.get_fignums()) > 0
        plt.close('all')

    def test_plot_with_none_predictions(self):
        """Test that function returns early with None predictions"""
        actual = np.random.randn(50)

        # Should not raise error
        plot_results(None, actual)

        # Should not create a figure
        assert len(plt.get_fignums()) == 0

    def test_plot_with_perfect_correlation(self):
        """Test plot with perfect correlation"""
        predictions = np.array([1, 2, 3, 4, 5])
        actual = predictions.copy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plot_results(predictions, actual, title="Perfect")

        assert len(plt.get_fignums()) > 0
        plt.close('all')

    def test_plot_with_no_correlation(self):
        """Test plot with no correlation"""
        np.random.seed(42)
        predictions = np.random.randn(50)
        actual = np.random.randn(50)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plot_results(predictions, actual, title="Random")

        assert len(plt.get_fignums()) > 0
        plt.close('all')
