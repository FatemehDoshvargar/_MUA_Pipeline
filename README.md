# A Configurable Pipeline for Mass Univariate Aggregation (MUA) Methods

## 1. A Configurable Pipeline for Mass Univariate Aggregation Methods

We present a unified, flexible, and accessible configurable pipeline that can be used for implementing CPM (Shen et al., 2017), PNRS (Byington et al., 2023), and many new MUA configurations facilitated by user-specified parameters. The pipeline’s core is built around two classes—**FeatureVectorizer** and **MUA**—both of which are engineered to be fully compatible with the scikit-learn ecosystem.

## 2. The Implementation Details

### 2.1 FeatureVectorizer Class
The FeatureVectorizer class helps to handle both 2D and 3D input data. In this manner, our configurable pipeline can be used to apply MUA methods on neuroimaging data (connectivity matrices) and any other feature-outcome data.

### 2.2 MUA Class
The MUA class is our main Python class that extends scikit-learn's BaseEstimator and TransformerMixin classes to ensure compatibility with standard machine learning pipelines; the MUA class operates through three steps:

1. **Feature Selection and Organization:** Determines which features are included and how they are organized (e.g., split into positive/negative networks or combined)
2. **Feature Weighting:** Specifies how selected features are weighted (e.g., binary, correlation-based, or regression-derived weights)
3. **Feature Aggregation:** Defines how weighted features are combined into predictive scores (e.g., sum or mean aggregation)

### 2.2.1 MUA Class’s Parameters
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
| | `'regression'` | feature-specific regression coefficients |
| **correlation_type** | `'pearson'` | Linear correlation |
| | `'spearman'` | Rank-based correlation |
| **feature_aggregation** | `'sum'` | The sum of the weighted features |
| | `'mean'` | The mean of the weighted features |
| **standardize_scores** | `True` | Z-score normalize final scores |
| | `False` | Keep raw scores |

### 2.3 Prediction Strategy
Our configurable pipeline facilitates the use of the various regression options available in scikit-learn. The following section demonstrates the practical application of this approach. While LinearRegression() is used here as an example, the pipeline supports the integration of any alternative Scikit-learn regression method.

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
## 3. Reference and Documentation

For comprehensive details, theoretical background, and extensive explanations regarding the use and validation of this pipeline, please refer to our full paper:

**Mass Univariate Aggregation Methods for Machine Learning in Neuroscience**
**DOI:** [https://doi.org/10.5281/zenodo.18436701]

### References
* Shen, X., Finn, E. S., Scheinost, D., Rosenberg, M. D., Chun, M. M., Papademetris, X., & Constable, R. T. (2017). Using connectome-based predictive modeling to predict individual behavior from brain connectivity. *Nature Protocols*, 12(3), 506-518.
* Byington, N. E., Schatza, J., & He, C. W. (2023). Predicting Individual Differences via Mass Univariate Aggregation: The Predictive Network Response Scale (PNRS).
### Software References
* YaleMRRC. Connectome-based Predictive Modeling (CPM) [Code repository]. https://github.com/YaleMRRC/CPM
* DCAN-Labs. BWAS: Polyneuro Risk Score [Code repository]. https://github.com/DCAN-Labs/BWAS
