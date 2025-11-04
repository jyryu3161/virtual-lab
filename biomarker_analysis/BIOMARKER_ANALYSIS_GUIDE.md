# Biomarker Analysis Implementation Guide

## Overview

This document provides a comprehensive guide to the biomarker analysis implementation added to the Virtual Lab platform. The biomarker analysis module enables identification of prognostic gene biomarkers from gene expression data with survival outcomes.

## Implementation Summary

### Files Created

```
virtual-lab/
├── biomarker_analysis/
│   ├── scripts/
│   │   ├── select_marker_genes.py       # Main analysis script (800+ lines)
│   │   └── example_usage.py             # Simple usage example
│   ├── tutorial_biomarker_selection.ipynb  # Comprehensive Jupyter tutorial
│   ├── requirements.txt                 # Python dependencies
│   ├── README.md                        # Module documentation
│   └── BIOMARKER_ANALYSIS_GUIDE.md      # This file
├── pixi.toml                            # Pixi package manager configuration
├── README.md                            # Updated main README
└── .gitignore                           # Updated to exclude result directories
```

## Core Features

### 1. Multiple Statistical Methods

The implementation includes four complementary statistical approaches:

#### A. Cox Proportional Hazards Regression
- **Implementation**: `method_cox_regression()` in `MarkerGeneSelector` class
- **Purpose**: Univariate survival analysis for each gene
- **Output**: Hazard ratios, coefficients, confidence intervals, p-values
- **Complexity**: O(n * m) where n = samples, m = genes

**Code Structure**:
```python
def method_cox_regression(self) -> pd.DataFrame:
    # For each gene:
    #   1. Create Cox model with gene expression as covariate
    #   2. Fit model using lifelines.CoxPHFitter
    #   3. Extract hazard ratio, p-value, confidence intervals
    #   4. Handle convergence failures gracefully
    # Return sorted DataFrame by p-value
```

#### B. Log-rank Test (Kaplan-Meier)
- **Implementation**: `method_logrank_test()` in `MarkerGeneSelector` class
- **Purpose**: Compare survival curves between high/low expression groups
- **Output**: Test statistics, p-values
- **Complexity**: O(n * m)

**Code Structure**:
```python
def method_logrank_test(self) -> pd.DataFrame:
    # For each gene:
    #   1. Split samples by median expression
    #   2. Create high/low expression groups
    #   3. Perform log-rank test using lifelines.logrank_test
    #   4. Calculate -log10(p-value) for visualization
    # Return sorted DataFrame by p-value
```

#### C. Differential Expression Analysis
- **Implementation**: `method_differential_expression()` in `MarkerGeneSelector` class
- **Purpose**: Identify genes differentially expressed between event/censored groups
- **Output**: Fold changes, test statistics, p-values
- **Complexity**: O(n * m)

**Code Structure**:
```python
def method_differential_expression(self) -> pd.DataFrame:
    # For each gene:
    #   1. Split samples by outcome (event vs censored)
    #   2. Perform Mann-Whitney U test (non-parametric)
    #   3. Calculate log fold change
    #   4. Compute statistics
    # Return sorted DataFrame by p-value
```

#### D. Elastic Net Cox Regression
- **Implementation**: `method_elasticnet_cox()` in `MarkerGeneSelector` class
- **Purpose**: Multivariate feature selection with L1/L2 regularization
- **Output**: Non-zero coefficients for selected genes
- **Complexity**: O(k * n * m) where k = CV folds

**Code Structure**:
```python
def method_elasticnet_cox(self) -> pd.DataFrame:
    # 1. Standardize features
    # 2. Cross-validation to find optimal alpha:
    #    - Test multiple alpha values (logspace)
    #    - K-fold cross-validation for each alpha
    #    - Select alpha with best CV score
    # 3. Fit final model with best alpha
    # 4. Extract non-zero coefficients
    # Return sorted DataFrame by absolute coefficient value
```

### 2. Consensus Gene Selection

The consensus approach combines results from multiple methods to identify robust biomarkers.

**Implementation**: `get_consensus_genes()` in `MarkerGeneSelector` class

```python
def get_consensus_genes(self, top_n: Optional[int] = None) -> pd.DataFrame:
    # 1. Extract top N genes from each method
    # 2. Count how many methods selected each gene
    # 3. Track which methods selected each gene
    # 4. Sort by number of methods (descending)
    # Return DataFrame with genes, n_methods, methods
```

**Logic**:
- Genes appearing in 4/4 methods: Most robust candidates
- Genes appearing in 3/4 methods: Strong candidates
- Genes appearing in 2/4 methods: Moderate candidates
- Genes appearing in 1/4 methods: Method-specific findings

### 3. Comprehensive Visualization

The visualization system generates a multi-page PDF with various plots.

**Implementation**: `visualize_results()` and helper methods

#### Plots Generated:

1. **Volcano Plots** (`_plot_volcano_cox()`, `_plot_volcano_differential()`)
   - X-axis: Effect size (coefficient or fold change)
   - Y-axis: -log10(p-value)
   - Highlights significant genes

2. **Top Genes Bar Plots** (`_plot_top_genes()`)
   - Shows top 20 genes from each method
   - Sorted by significance or coefficient magnitude

3. **Kaplan-Meier Curves** (`_plot_km_curves()`)
   - 3×3 grid of KM plots for top 9 consensus genes
   - High vs. low expression groups
   - Includes log-rank p-values

4. **Expression Heatmap** (`_plot_heatmap()`)
   - Top 30 consensus genes
   - Samples sorted by survival time
   - Color-coded by outcome

5. **Consensus Plot** (`_plot_consensus()`)
   - Bar chart showing gene overlap between methods
   - Counts genes by number of methods

### 4. Command-Line Interface

**Implementation**: Uses `typed-argument-parser` (tap) library

```python
class Args(Tap):
    """Argument parser for marker gene selection"""
    input_file: str = "Example_TCGA_TNBC_data.csv"
    output_dir: str = "biomarker_results"
    n_top_genes: int = 50
    p_value_threshold: float = 0.05
    cv_folds: int = 5
    methods: List[str] = ["cox", "logrank", "differential", "elasticnet"]
    visualization: bool = True
    random_seed: int = 42
```

**Usage**:
```bash
python select_marker_genes.py \
    --input_file data.csv \
    --output_dir results \
    --n_top_genes 50 \
    --methods cox logrank \
    --visualization
```

### 5. Result Export System

**Implementation**: `save_results()` in `MarkerGeneSelector` class

**Files Generated**:
- `consensus_genes.csv`: Ranked consensus genes
- `cox_results.csv`: Full Cox regression results
- `logrank_results.csv`: Full log-rank test results
- `differential_results.csv`: Full differential expression results
- `elasticnet_results.csv`: Elastic Net coefficients
- `top_genes_all_methods.csv`: Combined top genes
- `marker_gene_analysis.pdf`: All visualizations

## Dependencies

### Core Libraries

1. **pandas** (>=1.3.0)
   - Data manipulation and CSV I/O
   - Used for: Loading data, organizing results

2. **numpy** (>=1.20.0)
   - Numerical operations
   - Used for: Statistical calculations, array operations

3. **scipy** (>=1.7.0)
   - Scientific computing
   - Used for: Mann-Whitney U test (`stats.mannwhitneyu`)

4. **matplotlib** (>=3.4.0)
   - Plotting backend
   - Used for: All visualizations, PDF generation

5. **seaborn** (>=0.11.0)
   - Statistical visualization
   - Used for: Enhanced plots, color palettes, heatmaps

### Survival Analysis Libraries

6. **lifelines** (>=0.27.0)
   - Survival analysis toolkit
   - Used for:
     - `CoxPHFitter`: Cox proportional hazards regression
     - `KaplanMeierFitter`: Kaplan-Meier survival curves
     - `logrank_test`: Log-rank test for survival curves

7. **scikit-survival** (>=0.21.0)
   - Machine learning for survival analysis
   - Used for:
     - `CoxnetSurvivalAnalysis`: Elastic Net Cox regression
     - `Surv`: Survival data structure

### Machine Learning Libraries

8. **scikit-learn** (>=1.0.0)
   - Machine learning utilities
   - Used for:
     - `StandardScaler`: Feature standardization
     - `KFold`: Cross-validation

9. **statsmodels** (>=0.13.0)
   - Statistical modeling
   - Used for: Additional statistical tests (optional)

### Utility Libraries

10. **typed-argument-parser** (>=1.7.0)
    - Type-safe CLI argument parsing
    - Used for: Command-line interface

11. **tqdm** (>=4.62.0)
    - Progress bars
    - Used for: Showing analysis progress (optional)

## Installation Methods

### Method 1: Pixi (Recommended)

Pixi provides a complete, reproducible environment with all dependencies.

**Configuration** (`pixi.toml`):
```toml
[dependencies]
python = ">=3.10,<3.14"
pandas = "*"
numpy = "*"
scipy = "*"
matplotlib = "*"
seaborn = "*"
lifelines = "*"
scikit-survival = "*"
scikit-learn = "*"
statsmodels = "*"
typed-argument-parser = "*"
# ... other dependencies

[tasks]
biomarker-analysis = "python biomarker_analysis/scripts/select_marker_genes.py ..."
biomarker-tutorial = "jupyter notebook biomarker_analysis/tutorial_biomarker_selection.ipynb"
```

**Installation**:
```bash
curl -fsSL https://pixi.sh/install.sh | bash  # Install pixi
git clone https://github.com/zou-group/virtual_lab.git
cd virtual_lab
pixi install  # Install all dependencies
pixi shell    # Activate environment
```

**Benefits**:
- Single command installation
- Reproducible environments
- Cross-platform compatibility
- Fast dependency resolution

### Method 2: Conda/Mamba

**Installation**:
```bash
conda create -y -n virtual_lab python=3.12
conda activate virtual_lab
pip install virtual-lab
pip install lifelines scikit-survival pandas scipy matplotlib seaborn
```

### Method 3: Pip + requirements.txt

**Installation**:
```bash
pip install -r biomarker_analysis/requirements.txt
```

## Data Format Specification

### Input CSV Structure

**Required Columns**:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `sample` | String | Unique sample/patient identifier | "TCGA-A7-A0CE-01" |
| `OS` | Binary (0/1) | Overall survival event (0=censored, 1=event/death) | 0, 1 |
| `OS.year` | Float | Overall survival time in years | 2.94, 11.10 |
| `GENE1`, `GENE2`, ... | Float | Gene expression values (typically log-transformed) | 5.46, 9.45, ... |

**Example Data**:
```csv
sample,OS,OS.year,A1BG,AAAS,AAGAB,AAK1,AAMP,AASDH,ABAT,...
TCGA-A7-A0CE-01,0,2.94246575342466,5.4618,9.4485,10.3657,9.8236,9.8461,7.915,8.1825,...
TCGA-AR-A0U1-01,0,11.1013698630137,6.0017,9.5988,10.5184,9.9634,11.0425,7.6319,6.5904,...
```

### Output CSV Files

#### 1. consensus_genes.csv
```csv
gene,n_methods,methods
BRCA1,4,"cox, logrank, differential, elasticnet"
TP53,4,"cox, logrank, differential, elasticnet"
MYC,3,"cox, logrank, elasticnet"
```

#### 2. cox_results.csv
```csv
gene,coef,HR,HR_95CI_lower,HR_95CI_upper,p_value,log10_p
BRCA1,0.523,1.687,1.234,2.308,0.0012,2.921
TP53,0.412,1.510,1.145,1.992,0.0034,2.468
```

#### 3. logrank_results.csv
```csv
gene,test_statistic,p_value,log10_p
BRCA1,8.234,0.0041,2.387
TP53,6.891,0.0087,2.061
```

#### 4. differential_results.csv
```csv
gene,mean_event,mean_censored,log_FC,U_statistic,p_value,log10_p
BRCA1,7.234,6.512,0.722,1234.5,0.0023,2.638
TP53,8.123,7.456,0.667,1189.2,0.0045,2.347
```

#### 5. elasticnet_results.csv
```csv
gene,coefficient,abs_coefficient
BRCA1,0.234,0.234
TP53,0.189,0.189
MYC,-0.156,0.156
```

## Algorithm Details

### Cox Regression Algorithm

**Input**: Gene expression matrix X (n×m), survival times T (n×1), events E (n×1)

**For each gene i in 1..m**:
1. Create DataFrame: `{time: T, event: E, gene_expr: X[:, i]}`
2. Initialize Cox model: `cph = CoxPHFitter()`
3. Fit model: `cph.fit(df, duration_col='time', event_col='event')`
4. Extract results:
   - Coefficient: β
   - Hazard ratio: HR = exp(β)
   - P-value from likelihood ratio test
   - 95% CI: exp(β ± 1.96 × SE(β))
5. Handle failures: Set p-value = 1.0 if convergence fails

**Output**: DataFrame sorted by p-value

**Time Complexity**: O(n × m × k) where k = iterations to convergence (~10-50)

### Log-rank Test Algorithm

**Input**: Gene expression matrix X (n×m), survival times T (n×1), events E (n×1)

**For each gene i in 1..m**:
1. Calculate median: `median_i = median(X[:, i])`
2. Split samples:
   - High: samples where `X[:, i] > median_i`
   - Low: samples where `X[:, i] ≤ median_i`
3. Get survival data:
   - `T_high, E_high` for high expression group
   - `T_low, E_low` for low expression group
4. Perform log-rank test:
   - Calculate observed vs. expected events at each time point
   - Compute test statistic: χ² ~ χ²(1)
   - P-value from chi-square distribution
5. Store: test statistic, p-value

**Output**: DataFrame sorted by p-value

**Time Complexity**: O(n log n × m) due to sorting for each gene

### Differential Expression Algorithm

**Input**: Gene expression matrix X (n×m), events E (n×1)

**For each gene i in 1..m**:
1. Split by outcome:
   - Event group: `X_event = X[E == 1, i]`
   - Censored group: `X_censored = X[E == 0, i]`
2. Compute statistics:
   - Mean event: `μ_event = mean(X_event)`
   - Mean censored: `μ_censored = mean(X_censored)`
   - Log fold change: `log_FC = μ_event - μ_censored`
3. Perform Mann-Whitney U test:
   - Rank all values
   - Calculate U statistic
   - P-value from U distribution
4. Store: means, fold change, U statistic, p-value

**Output**: DataFrame sorted by p-value

**Time Complexity**: O(n log n × m) due to ranking

### Elastic Net Cox Algorithm

**Input**: Gene expression matrix X (n×m), survival data (T, E)

**Step 1: Cross-validation for alpha selection**
```
alphas = logspace(-4, 1, 50)
For each alpha in alphas:
    cv_scores = []
    For each fold in K-fold CV:
        1. Split data: (X_train, y_train), (X_val, y_val)
        2. Standardize: X_train_scaled, X_val_scaled
        3. Fit model: CoxnetSurvivalAnalysis(alpha=alpha, l1_ratio=0.5)
        4. Score on validation set
        5. Append score to cv_scores
    Store mean(cv_scores) for this alpha
Select best_alpha = argmax(mean(cv_scores))
```

**Step 2: Final model fitting**
```
1. Standardize full dataset: X_scaled
2. Fit final model with best_alpha
3. Extract coefficients: β (many will be zero due to L1 penalty)
4. Return genes with non-zero coefficients
```

**Output**: DataFrame with genes and coefficients, sorted by |β|

**Time Complexity**: O(k × n × m × log(m)) where k = CV folds

## Tutorial Notebook Structure

The Jupyter tutorial (`tutorial_biomarker_selection.ipynb`) provides a comprehensive walkthrough:

### Section Breakdown

1. **Introduction & Objectives** (Markdown)
   - Overview of biomarker analysis
   - Learning objectives
   - Dataset description

2. **Environment Setup** (Code)
   - Import libraries
   - Configure settings
   - Test imports

3. **Data Loading & Exploration** (Code + Visualization)
   - Load CSV data
   - Examine structure
   - Survival outcome distribution
   - Overall Kaplan-Meier curve

4. **Method 1: Cox Regression** (Code + Explanation)
   - Theoretical background
   - Example with few genes
   - Interpretation of hazard ratios

5. **Method 2: Log-rank Test** (Code + Visualization)
   - Kaplan-Meier curves for high/low groups
   - Log-rank test interpretation
   - Example plots

6. **Method 3: Differential Expression** (Code + Visualization)
   - Mann-Whitney U test
   - Fold change calculation
   - Boxplots comparing groups

7. **Method 4: Elastic Net** (Code + Theory)
   - Regularization concepts
   - Cross-validation
   - Feature selection

8. **Complete Analysis** (Code)
   - Run all methods on full dataset
   - Can be run from Python or command line

9. **Results Exploration** (Code + Visualization)
   - Load saved results
   - Compare methods
   - Visualize overlap

10. **Detailed Biomarker Analysis** (Code + Interpretation)
    - Top consensus genes
    - Detailed statistics for each gene
    - Kaplan-Meier plots

11. **Export & Summary** (Code)
    - Create summary tables
    - Export for further analysis

12. **Conclusion** (Markdown)
    - Summary of findings
    - Next steps
    - References

### Interactive Features

- **Executable cells**: All code can be run step-by-step
- **Visualizations**: Inline plots with explanations
- **Markdown cells**: Detailed explanations and interpretations
- **Modularity**: Can run full analysis or individual methods

## Usage Examples

### Example 1: Basic Usage

```bash
python select_marker_genes.py \
    --input_file Example_TCGA_TNBC_data.csv \
    --output_dir results \
    --n_top_genes 50 \
    --visualization
```

### Example 2: Specific Methods Only

```bash
python select_marker_genes.py \
    --input_file data.csv \
    --output_dir results \
    --methods cox logrank \
    --n_top_genes 100
```

### Example 3: Custom P-value Threshold

```bash
python select_marker_genes.py \
    --input_file data.csv \
    --output_dir results \
    --p_value_threshold 0.01 \
    --n_top_genes 30
```

### Example 4: Using Pixi Tasks

```bash
# Quick test (fast, only 2 methods)
pixi run test-biomarker

# Full analysis
pixi run biomarker-analysis

# Interactive tutorial
pixi run biomarker-tutorial
```

### Example 5: Python API

```python
from select_marker_genes import MarkerGeneSelector, Args

# Configure
args = Args(
    input_file="my_data.csv",
    output_dir="my_results",
    n_top_genes=50,
    methods=["cox", "elasticnet"]
)

# Run analysis
selector = MarkerGeneSelector(args)
selector.load_data()
selector.method_cox_regression()
selector.method_elasticnet_cox()

# Get consensus
consensus = selector.get_consensus_genes()
print(consensus.head(20))

# Save and visualize
selector.save_results(args.output_dir)
selector.visualize_results(args.output_dir)
```

## Performance Considerations

### Benchmark (Example TCGA TNBC Dataset)

- **Dataset**: 144 samples, ~20,000 genes
- **Hardware**: Modern laptop (8 cores, 16GB RAM)

| Method | Time | Memory |
|--------|------|--------|
| Cox regression | ~1-2 min | ~500 MB |
| Log-rank test | ~1-2 min | ~500 MB |
| Differential expression | ~30 sec | ~300 MB |
| Elastic Net (CV) | ~2-5 min | ~1 GB |
| Visualization | ~30 sec | ~200 MB |
| **Total** | **~5-10 min** | **~1.5 GB peak** |

### Scalability

For larger datasets:

| Dataset Size | Genes | Samples | Expected Time |
|--------------|-------|---------|---------------|
| Small | 1,000 | 100 | ~30 sec |
| Medium | 10,000 | 200 | ~3-5 min |
| Large | 20,000 | 500 | ~10-20 min |
| Very Large | 50,000 | 1,000 | ~1-2 hours |

**Bottlenecks**:
- Cox regression convergence (can be slow for some genes)
- Elastic Net cross-validation (scales with CV folds)

**Optimization strategies**:
- Run methods individually
- Use multiprocessing for gene-level analyses
- Pre-filter low-variance genes
- Reduce CV folds for Elastic Net

## Troubleshooting

### Common Issues

1. **ConvergenceWarning in Cox regression**
   - **Cause**: Some genes have numerical issues
   - **Solution**: Automatically handled (p-value = 1.0)
   - **Impact**: Minimal, these genes are likely not significant

2. **Memory errors with large datasets**
   - **Cause**: All gene data loaded into memory
   - **Solution**: Process in batches or filter genes first
   - **Alternative**: Use chunked reading with pandas

3. **Slow Elastic Net convergence**
   - **Cause**: Many features, cross-validation
   - **Solution**: Reduce `cv_folds` or `n_alphas`
   - **Alternative**: Use faster methods only

4. **Empty consensus genes**
   - **Cause**: Different methods select completely different genes
   - **Solution**: Increase `n_top_genes` or lower `p_value_threshold`
   - **Interpretation**: May indicate weak/noisy signal

## Testing

### Manual Testing Checklist

- [ ] Script runs without errors: `python select_marker_genes.py --help`
- [ ] Example data loads correctly
- [ ] Each method produces results
- [ ] Consensus genes are computed
- [ ] Results are saved to CSV files
- [ ] Visualizations are generated
- [ ] Tutorial notebook opens in Jupyter
- [ ] Pixi commands work: `pixi run test-biomarker`

### Unit Tests (Future Work)

Suggested test cases:
```python
def test_load_data():
    """Test data loading with example file"""

def test_cox_regression():
    """Test Cox regression with small dataset"""

def test_logrank():
    """Test log-rank test with known outcome"""

def test_differential_expression():
    """Test DE analysis with synthetic data"""

def test_consensus():
    """Test consensus gene selection logic"""
```

## Future Enhancements

### Potential Additions

1. **Additional Methods**
   - Random Survival Forests
   - Gradient Boosting for survival
   - Deep learning survival models

2. **Pathway Analysis**
   - Gene set enrichment analysis (GSEA)
   - Pathway over-representation
   - Network analysis

3. **External Validation**
   - Cross-dataset validation
   - Meta-analysis across cohorts

4. **Interactive Dashboard**
   - Streamlit or Dash web interface
   - Real-time parameter tuning
   - Interactive plots

5. **Performance Optimization**
   - Multiprocessing for parallel analysis
   - Caching intermediate results
   - GPU acceleration for ML methods

6. **Clinical Integration**
   - Risk score calculation
   - Nomogram generation
   - Treatment recommendation

## References

### Key Papers

1. **Cox Proportional Hazards**
   - Cox, D. R. (1972). "Regression models and life-tables." *Journal of the Royal Statistical Society*.

2. **Kaplan-Meier & Log-rank**
   - Kaplan, E. L., & Meier, P. (1958). "Nonparametric estimation from incomplete observations." *JASA*.

3. **Elastic Net**
   - Zou, H., & Hastie, T. (2005). "Regularization and variable selection via the elastic net." *Journal of the Royal Statistical Society*.

4. **Survival Analysis with R/Python**
   - Therneau, T. M., & Grambsch, P. M. (2000). *Modeling Survival Data: Extending the Cox Model*.
   - Davidson-Pilon, C. (2019). *lifelines: survival analysis in Python*.

### Software Documentation

- **lifelines**: https://lifelines.readthedocs.io/
- **scikit-survival**: https://scikit-survival.readthedocs.io/
- **pandas**: https://pandas.pydata.org/docs/
- **pixi**: https://pixi.sh/latest/

## Contact & Support

For issues, questions, or contributions:

- **GitHub Issues**: https://github.com/zou-group/virtual-lab/issues
- **Documentation**: See README files in each directory
- **Tutorial**: `biomarker_analysis/tutorial_biomarker_selection.ipynb`

## License

This biomarker analysis module is part of the Virtual Lab project and is licensed under the MIT License.

---

**Last Updated**: 2025-11-04
**Version**: 1.0.0
**Author**: Virtual Lab Development Team
