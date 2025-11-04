# Biomarker Analysis for Prognosis Prediction

This module provides a comprehensive toolkit for identifying prognostic biomarker genes from gene expression data with survival outcomes.

## Overview

The biomarker analysis pipeline implements multiple statistical methods to identify genes associated with patient survival, including:

1. **Cox Proportional Hazards Regression** - Univariate survival analysis for each gene
2. **Log-rank Test** - Kaplan-Meier survival curve comparison (high vs. low expression)
3. **Differential Expression Analysis** - Mann-Whitney U test between event and censored groups
4. **Elastic Net Cox Regression** - Regularized multivariate survival analysis with feature selection

## Directory Structure

```
biomarker_analysis/
├── scripts/
│   └── select_marker_genes.py    # Main analysis script
├── tutorial_biomarker_selection.ipynb  # Interactive tutorial
└── README.md                      # This file
```

## Quick Start

### Method 1: Using YAML Configuration (Recommended) 🆕

```bash
cd biomarker_analysis

# Create your config file from the example
cp config_example.yaml my_analysis.yaml

# Edit the config file to set your parameters
# vim my_analysis.yaml

# Run analysis with config
python scripts/run_with_config.py my_analysis.yaml

# Dry run (validate config without running)
python scripts/run_with_config.py my_analysis.yaml --dry-run
```

**Why use YAML config?**
- ✅ All parameters in one place
- ✅ Easy to reproduce experiments
- ✅ Version control your analysis settings
- ✅ Share configurations with collaborators
- ✅ No need to remember command-line arguments

See [Configuration Guide](#configuration-guide) for detailed parameter descriptions.

### Method 2: Using Pixi

```bash
# From the virtual-lab root directory

# Run complete analysis
pixi run biomarker-analysis

# Launch interactive tutorial
pixi run biomarker-tutorial

# Quick test (only Cox and Log-rank methods)
pixi run test-biomarker
```

### Method 3: Command Line

```bash
cd biomarker_analysis/scripts

python select_marker_genes.py \
    --input_file ../../Example_TCGA_TNBC_data.csv \
    --output_dir ../../biomarker_results \
    --n_top_genes 50 \
    --p_value_threshold 0.05 \
    --methods cox logrank differential elasticnet \
    --visualization
```

### Method 4: Jupyter Notebook

```bash
jupyter notebook tutorial_biomarker_selection.ipynb
```

Or from Python:

```python
from select_marker_genes import MarkerGeneSelector, Args

args = Args(
    input_file="../../Example_TCGA_TNBC_data.csv",
    output_dir="../../biomarker_results",
    n_top_genes=50,
    methods=["cox", "logrank", "differential", "elasticnet"]
)

selector = MarkerGeneSelector(args)
selector.load_data()
selector.method_cox_regression()
selector.method_logrank_test()
selector.method_differential_expression()
selector.method_elasticnet_cox()

consensus = selector.get_consensus_genes()
selector.save_results(args.output_dir)
selector.visualize_results(args.output_dir)
```

## Input Data Format

The input file should be a CSV with the following structure:

| Column | Description | Type |
|--------|-------------|------|
| `sample` | Sample/patient identifier | String |
| `OS` | Overall survival event (0 = censored, 1 = event/death) | Binary (0/1) |
| `OS.year` | Overall survival time in years | Numeric |
| `GENE1`, `GENE2`, ... | Gene expression values (typically log-transformed) | Numeric |

### Example

```csv
sample,OS,OS.year,A1BG,AAAS,AAGAB,...
TCGA-A7-A0CE-01,0,2.94,5.46,9.45,10.37,...
TCGA-AR-A0U1-01,0,11.10,6.00,9.60,10.52,...
TCGA-E2-A1II-01,0,2.81,5.89,9.95,9.82,...
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input_file` | str | `Example_TCGA_TNBC_data.csv` | Path to input CSV file |
| `--output_dir` | str | `biomarker_results` | Directory to save results |
| `--n_top_genes` | int | `50` | Number of top genes to select |
| `--p_value_threshold` | float | `0.05` | P-value threshold for significance |
| `--methods` | list | All methods | Methods to use: `cox`, `logrank`, `differential`, `elasticnet` |
| `--cv_folds` | int | `5` | Number of cross-validation folds (for Elastic Net) |
| `--visualization` | bool | `True` | Generate visualization plots |
| `--random_seed` | int | `42` | Random seed for reproducibility |

## Output Files

The analysis generates the following files in the output directory:

### Results Files

- **`consensus_genes.csv`**: Genes ranked by how many methods selected them
  - Columns: `gene`, `n_methods`, `methods`

- **`cox_results.csv`**: Cox regression results for all genes
  - Columns: `gene`, `coef`, `HR`, `HR_95CI_lower`, `HR_95CI_upper`, `p_value`, `log10_p`

- **`logrank_results.csv`**: Log-rank test results
  - Columns: `gene`, `test_statistic`, `p_value`, `log10_p`

- **`differential_results.csv`**: Differential expression results
  - Columns: `gene`, `mean_event`, `mean_censored`, `log_FC`, `U_statistic`, `p_value`, `log10_p`

- **`elasticnet_results.csv`**: Elastic Net coefficients
  - Columns: `gene`, `coefficient`, `abs_coefficient`

- **`top_genes_all_methods.csv`**: Combined top genes from all methods

- **`top_biomarkers_summary.csv`**: Comprehensive summary (generated in notebook)

### Visualization File

- **`marker_gene_analysis.pdf`**: Multi-page PDF containing:
  - Volcano plots (Cox regression and differential expression)
  - Top genes bar plots for each method
  - Kaplan-Meier curves for top consensus genes
  - Expression heatmap of top genes
  - Consensus gene overlap plot

## Statistical Methods Explained

### 1. Cox Proportional Hazards Regression

**Purpose**: Model the relationship between gene expression and survival time, accounting for censored data.

**Interpretation**:
- **Hazard Ratio (HR) > 1**: Higher expression → worse survival (risk factor)
- **Hazard Ratio (HR) < 1**: Higher expression → better survival (protective factor)
- **P-value**: Statistical significance of the association

**Use case**: Identify genes whose expression levels are continuously associated with survival risk.

### 2. Log-rank Test (Kaplan-Meier)

**Purpose**: Compare survival curves between high and low expression groups.

**Method**:
1. Split patients by median gene expression
2. Generate Kaplan-Meier survival curves for each group
3. Test if curves are significantly different (log-rank test)

**Interpretation**:
- **Low p-value**: Significant survival difference between groups
- Visualized with Kaplan-Meier curves

**Use case**: Identify genes that stratify patients into distinct prognostic groups.

### 3. Differential Expression Analysis

**Purpose**: Find genes with different expression between event and censored groups.

**Method**: Mann-Whitney U test (non-parametric)

**Interpretation**:
- **Positive log FC**: Higher expression in event (death) group
- **Negative log FC**: Higher expression in censored group
- **Low p-value**: Significant expression difference

**Use case**: Identify genes differentially expressed based on outcome, regardless of time.

### 4. Elastic Net Cox Regression

**Purpose**: Multivariate survival analysis with automatic feature selection.

**Method**:
- Combines L1 (Lasso) and L2 (Ridge) regularization
- Performs cross-validation to select optimal regularization parameter
- Returns sparse set of genes with non-zero coefficients

**Interpretation**:
- **Non-zero coefficient**: Gene selected by the model
- **Coefficient magnitude**: Importance of the gene

**Use case**: Build a sparse, predictive model using multiple genes simultaneously.

## Consensus Gene Selection

Genes that appear as significant across multiple methods are more likely to be robust biomarkers. The consensus approach:

1. Takes top N genes from each method
2. Counts how many methods selected each gene
3. Ranks genes by the number of methods

**Example Output**:

```
gene      n_methods  methods
BRCA1     4          cox, logrank, differential, elasticnet
TP53      4          cox, logrank, differential, elasticnet
MYC       3          cox, logrank, elasticnet
KRAS      2          cox, differential
```

## Example: TCGA TNBC Dataset

The included example dataset contains:

- **144 samples** from TCGA triple-negative breast cancer patients
- **~20,000 genes** with log-transformed expression values
- **Survival data**: Overall survival (OS) and survival time (OS.year)

### Expected Runtime

- **Cox regression**: ~1-2 minutes (20,000 genes)
- **Log-rank test**: ~1-2 minutes
- **Differential expression**: ~30 seconds
- **Elastic Net**: ~2-5 minutes (with cross-validation)

**Total**: ~5-10 minutes for complete analysis

## Troubleshooting

### ImportError: No module named 'lifelines'

Install required packages:

```bash
pip install lifelines scikit-survival statsmodels
```

Or use pixi:

```bash
pixi install
```

### ConvergenceWarning in Cox regression

Some genes may fail to converge. This is normal and these genes will be assigned p-value = 1.0.

### Memory issues with large datasets

For datasets with >100,000 genes:
- Run methods individually
- Use `--methods cox` to start
- Consider filtering low-variance genes first

### Visualization errors

Ensure matplotlib backend is properly configured:

```python
import matplotlib
matplotlib.use('Agg')  # For non-interactive backend
```

## Citation

If you use this biomarker analysis module, please cite the Virtual Lab paper:

Swanson, K., Wu, W., Bulaong, N.L. et al. The Virtual Lab of AI agents designs new SARS-CoV-2 nanobodies. *Nature* (2025). https://doi.org/10.1038/s41586-025-09442-9

## References

### Statistical Methods

- Cox, D. R. (1972). Regression models and life-tables. *Journal of the Royal Statistical Society*.
- Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from incomplete observations. *Journal of the American Statistical Association*.
- Simon, N., et al. (2011). Regularization paths for Cox's proportional hazards model via coordinate descent. *Journal of Statistical Software*.

### Survival Analysis Packages

- **lifelines**: https://lifelines.readthedocs.io/
- **scikit-survival**: https://scikit-survival.readthedocs.io/

### Dataset

- TCGA Network (2012). Comprehensive molecular portraits of human breast tumours. *Nature*.

## Methodology: How It Works

### Overall Approach

The biomarker discovery pipeline follows a **multi-method consensus approach** to identify robust prognostic genes:

```
Input Data → Preprocessing → Statistical Methods (4) → Consensus → Validation → Output
```

#### Step 1: Data Loading and Preprocessing
1. **Load gene expression matrix** with survival outcomes
2. **Quality control**:
   - Check for missing values
   - Verify survival data (time > 0, event in {0,1})
   - Filter low-variance genes (optional)
3. **Data structure**:
   - Rows = Patients/samples
   - Columns = Genes + survival columns
   - Values = Log-transformed expression (typically log2(TPM+1) or log2(FPKM+1))

#### Step 2: Statistical Analysis (4 Methods)

Each method addresses a different aspect of prognostic association:

| Method | Question Addressed | Output |
|--------|-------------------|--------|
| **Cox Regression** | Is gene expression continuously associated with survival risk? | Hazard Ratio, p-value |
| **Log-rank Test** | Do high/low expression groups have different survival curves? | Test statistic, p-value |
| **Differential Expression** | Is expression different between event and censored patients? | Fold change, p-value |
| **Elastic Net** | What minimal set of genes predicts survival together? | Coefficients |

**Why use multiple methods?**
- **Robustness**: Genes identified by multiple methods are more reliable
- **Complementarity**: Each method captures different aspects of prognostic association
- **Validation**: Cross-method validation reduces false positives

#### Step 3: Consensus Gene Selection

**Algorithm**:
1. Select top N genes from each method (e.g., N=50)
2. For each gene, count how many methods selected it (n_methods)
3. Rank genes by n_methods (higher = more robust)
4. Optionally weight methods by reliability

**Example**:
```
Gene A: Selected by Cox + Log-rank + Elastic Net = 3 methods → High confidence
Gene B: Selected by Cox only = 1 method → Lower confidence
```

#### Step 4: Multiple Testing Correction

**Problem**: Testing 20,000 genes leads to many false positives at α=0.05

**Solution**: Benjamini-Hochberg FDR correction
- Controls the **False Discovery Rate** (expected proportion of false positives)
- More powerful than Bonferroni correction
- Adjusted p-values (q-values) provided in results

#### Step 5: Visualization and Interpretation

**Automated visualizations**:
1. **Volcano plots**: Effect size vs. significance
2. **Kaplan-Meier curves**: Survival stratification for top genes
3. **Heatmaps**: Expression patterns of biomarker genes
4. **Forest plots**: Hazard ratios with confidence intervals

### Detailed Method Descriptions

#### Cox Proportional Hazards Regression

**Mathematical Model**:
```
h(t|X) = h₀(t) × exp(β × X)

where:
  h(t|X) = hazard at time t given gene expression X
  h₀(t) = baseline hazard
  β = regression coefficient
  HR = exp(β) = hazard ratio
```

**Implementation**:
```python
for each gene:
    1. Fit Cox model: survival ~ gene_expression
    2. Extract coefficient β and p-value
    3. Calculate HR = exp(β)
    4. Compute 95% confidence interval
    5. Adjust p-values (FDR correction)
```

**Interpretation**:
- **HR = 2.0**: 1 unit increase in expression → 2× higher risk
- **HR = 0.5**: 1 unit increase in expression → 50% lower risk
- **p < 0.05**: Statistically significant association

**Assumptions**:
- Proportional hazards (hazard ratio constant over time)
- Linear relationship between log(hazard) and expression
- Independent censoring

#### Log-rank Test with Kaplan-Meier

**Algorithm**:
```python
for each gene:
    1. Split patients by median expression:
       - High expression group (above median)
       - Low expression group (below median)
    2. Fit Kaplan-Meier survival curves for each group
    3. Perform log-rank test to compare curves
    4. Calculate p-value
```

**Test Statistic**:
```
χ² = Σ(Oᵢ - Eᵢ)² / Eᵢ

where:
  Oᵢ = observed events in group i
  Eᵢ = expected events in group i
```

**Advantages**:
- Non-parametric (no distribution assumptions)
- Intuitive visualization with survival curves
- Robust to outliers

**Disadvantages**:
- Dichotomizes continuous expression (loses information)
- Assumes proportional hazards

#### Differential Expression Analysis

**Algorithm**:
```python
for each gene:
    1. Divide patients into two groups:
       - Event group (OS = 1, death occurred)
       - Censored group (OS = 0, still alive)
    2. Perform Mann-Whitney U test:
       H₀: No difference in expression between groups
    3. Calculate fold change (log scale):
       log_FC = log₂(mean_event / mean_censored)
    4. Compute p-value and adjust for multiple testing
```

**Mann-Whitney U Test**:
- Non-parametric alternative to t-test
- Tests if distributions differ
- No normality assumption
- Robust to outliers

**Interpretation**:
- **log_FC > 0**: Gene upregulated in patients who died
- **log_FC < 0**: Gene downregulated in patients who died
- **p < 0.05**: Significant expression difference

**Note**: This method ignores survival time, only considers final outcome

#### Elastic Net Cox Regression

**Objective Function**:
```
min  -log_likelihood + λ × [(1-α)/2 × ||β||₂² + α × ||β||₁]
 β

where:
  λ = regularization strength (found by cross-validation)
  α = mixing parameter (0.5 = equal L1 and L2)
  ||β||₁ = L1 penalty (promotes sparsity)
  ||β||₂² = L2 penalty (shrinks coefficients)
```

**Algorithm**:
```python
1. Standardize gene expression data (mean=0, std=1)
2. For each λ in grid of values:
   a. Perform k-fold cross-validation
   b. Fit Elastic Net Cox on training folds
   c. Evaluate C-index on validation folds
3. Select λ that maximizes C-index
4. Refit model on all data with optimal λ
5. Extract non-zero coefficients
```

**Advantages**:
- **Multivariate**: Considers all genes simultaneously
- **Feature selection**: Sets irrelevant coefficients to zero
- **Regularization**: Prevents overfitting
- **Handles collinearity**: L2 penalty groups correlated genes

**Interpretation**:
- Non-zero coefficients = genes selected by the model
- Coefficient magnitude = importance
- Sign indicates direction (positive = risk, negative = protective)

### Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT DATA                                │
│  Gene Expression Matrix + Survival Outcomes (OS, OS.year)   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
                  ┌─────────────────┐
                  │  Preprocessing  │
                  │  - QC checks    │
                  │  - Filter genes │
                  └────────┬────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
┌────────────────┐ ┌──────────────┐ ┌─────────────────┐
│ Cox Regression │ │ Log-rank Test│ │ Differential Exp│
│  - Univariate  │ │ - KM curves  │ │ - Mann-Whitney  │
│  - HR, p-value │ │ - p-value    │ │ - FC, p-value   │
└───────┬────────┘ └──────┬───────┘ └────────┬────────┘
        │                 │                   │
        │                 ▼                   │
        │        ┌─────────────────┐          │
        │        │  Elastic Net    │          │
        │        │  - Multivariate │          │
        │        │  - CV selection │          │
        │        └────────┬────────┘          │
        │                 │                   │
        └────────┬────────┴────────┬──────────┘
                 │                 │
                 ▼                 ▼
         ┌────────────────────────────────┐
         │    Consensus Gene Selection     │
         │  - Count methods per gene       │
         │  - Rank by robustness          │
         │  - Weight methods (optional)   │
         └──────────────┬─────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │  Visualizations  │
              │  - Volcano plots │
              │  - KM curves     │
              │  - Heatmaps      │
              └─────────┬────────┘
                        │
                        ▼
                ┌───────────────┐
                │    OUTPUT     │
                │ - CSV results │
                │ - PDF figures │
                │ - Gene lists  │
                └───────────────┘
```

## Configuration Guide

The `config_example.yaml` file provides a comprehensive way to configure all analysis parameters:

### Key Configuration Sections

#### 1. Data Configuration
```yaml
data:
  input_file: "../Example_TCGA_TNBC_data.csv"
  columns:
    survival_time: "OS.year"
    survival_event: "OS"
  filters:
    min_expression_variance: 0.0
```

**Parameters**:
- `input_file`: Path to CSV with gene expression and survival data
- `survival_time`: Column name for survival time (must be numeric)
- `survival_event`: Column name for event indicator (0/1)
- `min_expression_variance`: Remove genes with variance below threshold

#### 2. Analysis Methods
```yaml
methods:
  enabled:
    - cox
    - logrank
    - differential
    - elasticnet

  cox:
    alpha: 0.05
    adjust_multiple_testing: true
    correction_method: "fdr_bh"
```

**Control which methods to run** and their parameters:
- Enable/disable methods by name
- Set significance thresholds (α)
- Configure multiple testing correction
- Method-specific parameters (e.g., l1_ratio for Elastic Net)

#### 3. Consensus Settings
```yaml
consensus:
  min_methods: 2          # Require at least 2 methods
  n_top_genes: 50         # Top genes per method
  method_weights:         # Optional weighting
    cox: 1.0
    elasticnet: 1.2       # Weight Elastic Net more
```

#### 4. Visualization
```yaml
visualization:
  enabled: true
  plots:
    volcano_plot: true
    kaplan_meier: true
    heatmap: true
  kaplan_meier:
    n_top_genes: 10       # Plot top 10 genes
    show_confidence: true
```

### Example Configurations

**Fast analysis (Cox only)**:
```yaml
methods:
  enabled:
    - cox
  cox:
    alpha: 0.01  # Stricter threshold
```

**Conservative (high stringency)**:
```yaml
consensus:
  min_methods: 3          # Require 3+ methods
  n_top_genes: 20         # Smaller top gene list
methods:
  cox:
    alpha: 0.001          # Very stringent
```

**Large dataset optimization**:
```yaml
data:
  filters:
    min_expression_variance: 0.1  # Filter low-variance genes
    remove_low_variance_genes: true
advanced:
  n_jobs: -1              # Use all CPUs
  low_memory_mode: true
```

## License

This module is part of the Virtual Lab project and is licensed under the MIT License.

## Contact

For questions or issues specific to biomarker analysis:
- Open an issue on GitHub
- Refer to the main Virtual Lab documentation
