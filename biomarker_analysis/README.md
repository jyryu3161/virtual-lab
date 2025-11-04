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

### Method 1: Using Pixi (Recommended)

```bash
# From the virtual-lab root directory

# Run complete analysis
pixi run biomarker-analysis

# Launch interactive tutorial
pixi run biomarker-tutorial

# Quick test (only Cox and Log-rank methods)
pixi run test-biomarker
```

### Method 2: Command Line

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

### Method 3: Jupyter Notebook

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

## License

This module is part of the Virtual Lab project and is licensed under the MIT License.

## Contact

For questions or issues specific to biomarker analysis:
- Open an issue on GitHub
- Refer to the main Virtual Lab documentation
