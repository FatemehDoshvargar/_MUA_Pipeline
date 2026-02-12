# A Configurable Pipeline for Mass Univariate Aggregation (MUA) Methods
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 1. A Configurable Pipeline for Mass Univariate Aggregation Methods

We present a unified, flexible, and accessible configurable pipeline that can be used for implementing CPM (Shen et al., 2017), PNRS (Byington et al., 2023), and many new MUA configurations facilitated by user-specified parameters. The pipeline’s core is built around two classes—**FeatureVectorizer** and **MUA**—both of which are engineered to be fully compatible with the scikit-learn ecosystem.

## 2. Installation

```bash
pip install git+https://github.com/neuroprismlab/_MUA_Pipeline.git
```

## 3. Project Structure

```
_MUA_Pipeline/
├── mua_pipeline/                    # Main package directory
│   ├── __init__.py                  # Package initialization
│   ├── core.py                      # Core classes: FeatureVectorizer and MUA
│   ├── preprocessing.py             # Data format auto-detection and missing data removal
│   └── visualization.py             # Plotting and visualization utilities
├── examples/
│   ├── CPM_PNRS_tutorial.ipynb      # Notebook walkthrough tutorial of CPM and PNRS with MUA
│   └── example_usage.py             # Example usage of MUA with a python script
├── validation/                      # Validation scripts and reference comparisons
│   ├── CPM_Validation.m             # MATLAB CPM validation
│   └── PNRS_Validation.m            # MATLAB PNRS validation
├── setup.py                         # Package installation configuration
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 4. The Implementation Details

### 4.1 FeatureVectorizer Class
The FeatureVectorizer class helps to handle both 2D and 3D input data. In this manner, our configurable pipeline can be used to apply MUA methods on neuroimaging data (connectivity matrices) and any other feature-outcome data.

### 4.2 MUA Class
The MUA class is our main Python class that extends scikit-learn's BaseEstimator and TransformerMixin classes to ensure compatibility with standard machine learning pipelines; the MUA class operates through three steps:

1. **Feature Selection and Organization:** Determines which features are included and how they are organized (e.g., split into positive/negative networks or combined)
2. **Feature Weighting:** Specifies how selected features are weighted (e.g., binary, correlation-based, or regression-derived weights)
3. **Feature Aggregation:** Defines how weighted features are combined into predictive scores (e.g., sum or mean aggregation)

### 4.2.1 MUA Class's Parameters
The MUA class employs seven configurable parameters organized across the three computational steps described above that enable flexible implementation of established methods while facilitating exploration of novel approaches.

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **split_by_sign** | `True` | Separate positive/negative features |
| | `False` | Keep all features together |
| **selection_method** | `'all'` | Use all features |
| | `'pvalue'` | Select features with p < α |
| | `'top_k'` | Select the top k features by absolute correlation |
| **selection_threshold** | `float (0, 1)` | 'pvalue' method: p-value threshold |
| | `integer` | 'top_k' method: number of features |
| | `N/A` | For 'all' method: ignore it |
| **weighting_method** | `'binary'` | ±1 based on correlation sign |
| | `'correlation'` | The strength of the correlation |
| | `'squared_correlation'` | r² preserving sign |
| | `'regression'` | Feature-specific regression coefficients |
| **correlation_type** | `'pearson'` | Linear correlation |
| | `'spearman'` | Rank-based correlation |
| **feature_aggregation** | `'sum'` | The sum of the weighted features |
| | `'mean'` | The mean of the weighted features |
| **standardize_scores** | `True` | Z-score normalize final scores |
| | `False` | Keep raw scores |

### 4.3 Prediction Strategy
Our configurable pipeline facilitates the use of the various regression options available in scikit-learn. The following section demonstrates the practical application of this approach. While LinearRegression() is used here as an example, the pipeline supports the integration of any alternative scikit-learn regression method.

```python
The_pipeline = Pipeline([
    ('vectorize', FeatureVectorizer()),
    ('mua', MUA(
        ...,
        ...,
    )),
    ('regressor', LinearRegression())  # Linear regression
])
```

## 5. Tutorial: Getting Started

See [examples](examples/) directory for specific examples of using the MUA pipeline.

### 5.1 Importing the Pipeline Components

```python
from mua_pipeline import FeatureVectorizer, MUA, preprocess, plot_results
```

### 5.2 Preprocessing Your Data
Before building a pipeline, you can use the **Preprocessing** module to handle data formatting and missing data removal. The `preprocess` function automatically detects the orientation of your data and converts it to the standard format expected by the pipeline:

* **3D connectivity matrices** are converted to `(n_subjects, n_regions, n_regions)` regardless of whether the input is `(n_regions, n_regions, n_subjects)`, `(n_regions, n_subjects, n_regions)`, or already in the standard format.
* **2D feature matrices** are converted to `(n_subjects, n_features)` even if provided as `(n_features, n_subjects)`.
* **Behavioral data** is similarly standardized.

```python
# Raw data — may have missing values and non-standard orientation
connectivity_matrices = ...   # e.g., shape (n_regions, n_regions, n_subjects)
behavioral_scores = ...       # e.g., shape (n_subjects,)

# Clean the data: auto-detects format, removes missing subjects, returns standard format
clean_connectivity, clean_behavioral, removed_indices = preprocess(
    connectivity_matrices,
    behavioral_scores,
    missing_strategy='any',   # Remove subjects with zeros, NaNs, or Infs
    verbose=True
)
```

The `missing_strategy` parameter controls what counts as missing data:

| Strategy | Removes subjects with |
| :--- | :--- |
| `'zero'` | Behavioral values equal to 0 |
| `'nan'` | NaN values |
| `'inf'` | Inf or -Inf values |
| `'any'` | Any of the above (default) |


### 5.3 Implementing CPM (Shen et al., 2017)
CPM uses binary weights, p-value-based feature selection, and splits features into positive and negative networks. A final linear regression maps the two network scores to the behavioral outcome.

```python
import numpy as np
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold

# Build the CPM pipeline
cpm_pipeline = Pipeline([
    ('vectorize', FeatureVectorizer()),
    ('mua', MUA(
        split_by_sign=True,            # Separate positive/negative networks
        selection_method='pvalue',     # p-value thresholding
        selection_threshold=0.05,      # p < 0.05
        weighting_method='binary',     # Binary weights (+1/−1)
        correlation_type='pearson',    # Pearson correlation
        feature_aggregation='sum',     # Sum of selected features
        standardize_scores=False
    )),
    ('regressor', LinearRegression())  # Final linear regression
])

# Cross-validation
cpm_scores = cross_val_score(cpm_pipeline, brain_data, behavior, cv=10)
cpm_predictions = cross_val_predict(cpm_pipeline, brain_data, behavior, cv=10)
print(f"CPM R² (10-fold CV): {cpm_scores.mean():.3f} ± {cpm_scores.std():.3f}")

# Evaluation
cpm_r, cpm_p = pearsonr(behavior, cpm_predictions)
mae = mean_absolute_error(behavior, cpm_predictions)
rmse = np.sqrt(mean_squared_error(behavior, cpm_predictions))
r2 = r2_score(behavior, cpm_predictions)
```

With `split_by_sign=True`, the MUA transformer outputs two columns — one for the positive network summary and one for the negative network summary. The `LinearRegression()` then fits on these two scores to predict the behavioral outcome.


### 5.4 Implementing PNRS (Byington et al., 2023)
PNRS uses regression-derived weights, includes all features, and produces a single combined score. The aggregated score is directly used as the prediction, so no downstream regressor is needed.

```python
# Build the PNRS pipeline
pnrs_pipeline = Pipeline([
    ('vectorize', FeatureVectorizer()),
    ('mua', MUA(
        split_by_sign=False,               # Single combined score
        selection_method='all',            # Use all features
        weighting_method='regression',     # Regression-derived weights
        feature_aggregation='sum',         # Sum of weighted features
        standardize_scores=True            # Z-score normalization
    ))
    # No regressor 
])

pnrs_scores = pnrs_pipeline.fit_transform(brain_data, behavior)

# Use scores directly as predictions
pnrs_predictions = pnrs_scores.flatten()

# Evaluation
pnrs_r, pnrs_p = pearsonr(behavior, pnrs_predictions)
mae = mean_absolute_error(behavior, pnrs_predictions)
rmse = np.sqrt(mean_squared_error(behavior, pnrs_predictions))
r2 = r2_score(behavior, pnrs_predictions)
```

### 5.5 Visualizing the Results
The **Visualization** module provides the `plot_results` function for visualizing predicted vs. observed behavioral scores:

```python
plot_results(cpm_predictions, behavioral_scores, title="CPM")
plot_results(pnrs_predictions, behavioral_scores, title="PNRS")
```

## 6. Reference and Documentation

For comprehensive details, theoretical background, and extensive explanations regarding the use and validation of this pipeline, please refer to our preprint:

**Mass Univariate Aggregation Methods for Machine Learning in Neuroscience**
**DOI:** [https://doi.org/10.5281/zenodo.18436701]

### References
* Shen X, Finn ES, Scheinost D, et al. Using connectome-based predictive modeling to predict individual behavior from brain connectivity. Nat Protoc. 2017;12(3):506-518. doi:10.1038/nprot.2016.178 
* Finn ES, Shen X, Scheinost D, et al. Functional connectome fingerprinting: identifying individuals using patterns of brain connectivity. Nat Neurosci. 2015;18(11):1664-1671. doi:10.1038/nn.4135 
* Byington N, Grimsrud G, Mooney MA, et al. Polyneuro risk scores capture widely distributed connectivity patterns of cognition. Dev Cogn Neurosci. 2023;60:101231. doi:10.1016/j.dcn.2023.101231 
* YaleMRRC. Connectome-based Predictive Modeling (CPM) [Code repository]. https://github.com/YaleMRRC/CPM
* DCAN-Labs. BWAS: Polyneuro Risk Score [Code repository]. https://github.com/DCAN-Labs/BWAS
