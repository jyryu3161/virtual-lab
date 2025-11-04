"""
Marker Gene Selection for Prognosis Prediction

This script implements multiple statistical methods for selecting
prognostic biomarker genes from gene expression data.

Methods:
1. Cox Proportional Hazards Model
2. Log-rank test (Kaplan-Meier survival analysis)
3. Differential expression analysis (Mann-Whitney U test)
4. Elastic Net Cox Regression with cross-validation

Author: Virtual Lab
Date: 2025-11-04
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Statistical and survival analysis
from scipy import stats
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Argument parsing
from tap import Tap


class Args(Tap):
    """Argument parser for marker gene selection"""

    input_file: str = "Example_TCGA_TNBC_data.csv"
    """Path to input CSV file with gene expression data"""

    output_dir: str = "biomarker_results"
    """Directory to save results"""

    n_top_genes: int = 50
    """Number of top genes to select"""

    p_value_threshold: float = 0.05
    """P-value threshold for significance"""

    cv_folds: int = 5
    """Number of cross-validation folds"""

    methods: List[str] = ["cox", "logrank", "differential", "elasticnet"]
    """Methods to use: cox, logrank, differential, elasticnet"""

    visualization: bool = True
    """Generate visualization plots"""

    random_seed: int = 42
    """Random seed for reproducibility"""


class MarkerGeneSelector:
    """
    A comprehensive class for selecting prognostic marker genes.

    This class implements multiple statistical approaches for identifying
    genes associated with patient survival outcomes.
    """

    def __init__(self, args: Args):
        self.args = args
        self.data = None
        self.gene_names = None
        self.results = {}

        # Set random seed
        np.random.seed(args.random_seed)

    def load_data(self) -> pd.DataFrame:
        """Load and prepare gene expression data"""
        print(f"Loading data from {self.args.input_file}...")

        # Load data
        self.data = pd.read_csv(self.args.input_file)

        # Remove quotes from column names if present
        self.data.columns = self.data.columns.str.replace('"', '')

        # Get gene names (exclude sample, OS, OS.year)
        self.gene_names = [col for col in self.data.columns
                          if col not in ['sample', 'OS', 'OS.year']]

        print(f"Loaded {len(self.data)} samples with {len(self.gene_names)} genes")
        print(f"Events: {self.data['OS'].sum()}, Censored: {(1-self.data['OS']).sum()}")

        return self.data

    def method_cox_regression(self) -> pd.DataFrame:
        """
        Univariate Cox Proportional Hazards regression for each gene.

        Returns:
            DataFrame with gene names, hazard ratios, p-values, and confidence intervals
        """
        print("\n=== Running Cox Proportional Hazards Analysis ===")

        results = []

        for i, gene in enumerate(self.gene_names):
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(self.gene_names)} genes...")

            # Prepare data for Cox model
            df_cox = pd.DataFrame({
                'time': self.data['OS.year'],
                'event': self.data['OS'],
                'gene_expr': self.data[gene]
            })

            try:
                # Fit Cox model
                cph = CoxPHFitter()
                cph.fit(df_cox, duration_col='time', event_col='event')

                # Extract results
                coef = cph.params_['gene_expr']
                hr = np.exp(coef)
                p_value = cph.summary['p']['gene_expr']
                ci_lower = np.exp(cph.confidence_intervals_['gene_expr']['95% lower-bound'])
                ci_upper = np.exp(cph.confidence_intervals_['gene_expr']['95% upper-bound'])

                results.append({
                    'gene': gene,
                    'coef': coef,
                    'HR': hr,
                    'HR_95CI_lower': ci_lower,
                    'HR_95CI_upper': ci_upper,
                    'p_value': p_value,
                    'log10_p': -np.log10(p_value) if p_value > 0 else np.inf
                })
            except Exception as e:
                # Handle convergence issues
                results.append({
                    'gene': gene,
                    'coef': np.nan,
                    'HR': np.nan,
                    'HR_95CI_lower': np.nan,
                    'HR_95CI_upper': np.nan,
                    'p_value': 1.0,
                    'log10_p': 0.0
                })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('p_value')

        print(f"Significant genes (p < {self.args.p_value_threshold}): "
              f"{(results_df['p_value'] < self.args.p_value_threshold).sum()}")

        self.results['cox'] = results_df
        return results_df

    def method_logrank_test(self) -> pd.DataFrame:
        """
        Log-rank test for survival difference between high/low expression groups.

        Returns:
            DataFrame with gene names, test statistics, and p-values
        """
        print("\n=== Running Log-rank Test (Kaplan-Meier) ===")

        results = []

        for i, gene in enumerate(self.gene_names):
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(self.gene_names)} genes...")

            # Split into high/low expression based on median
            median_expr = self.data[gene].median()
            high_expr = self.data[gene] > median_expr

            # Prepare data
            time_high = self.data.loc[high_expr, 'OS.year']
            event_high = self.data.loc[high_expr, 'OS']
            time_low = self.data.loc[~high_expr, 'OS.year']
            event_low = self.data.loc[~high_expr, 'OS']

            try:
                # Perform log-rank test
                result = logrank_test(
                    time_high, time_low,
                    event_high, event_low
                )

                results.append({
                    'gene': gene,
                    'test_statistic': result.test_statistic,
                    'p_value': result.p_value,
                    'log10_p': -np.log10(result.p_value) if result.p_value > 0 else np.inf
                })
            except Exception as e:
                results.append({
                    'gene': gene,
                    'test_statistic': np.nan,
                    'p_value': 1.0,
                    'log10_p': 0.0
                })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('p_value')

        print(f"Significant genes (p < {self.args.p_value_threshold}): "
              f"{(results_df['p_value'] < self.args.p_value_threshold).sum()}")

        self.results['logrank'] = results_df
        return results_df

    def method_differential_expression(self) -> pd.DataFrame:
        """
        Differential expression analysis between censored and event groups.

        Uses Mann-Whitney U test to identify genes with different expression
        levels between patients who had events vs. censored patients.

        Returns:
            DataFrame with gene names, test statistics, fold changes, and p-values
        """
        print("\n=== Running Differential Expression Analysis ===")

        results = []

        # Split by outcome
        event_mask = self.data['OS'] == 1

        for i, gene in enumerate(self.gene_names):
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(self.gene_names)} genes...")

            expr_event = self.data.loc[event_mask, gene]
            expr_censored = self.data.loc[~event_mask, gene]

            # Mann-Whitney U test (non-parametric)
            statistic, p_value = stats.mannwhitneyu(
                expr_event, expr_censored, alternative='two-sided'
            )

            # Calculate fold change (in log space, this is a difference)
            mean_event = expr_event.mean()
            mean_censored = expr_censored.mean()
            log_fc = mean_event - mean_censored

            results.append({
                'gene': gene,
                'mean_event': mean_event,
                'mean_censored': mean_censored,
                'log_FC': log_fc,
                'U_statistic': statistic,
                'p_value': p_value,
                'log10_p': -np.log10(p_value) if p_value > 0 else np.inf
            })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('p_value')

        print(f"Significant genes (p < {self.args.p_value_threshold}): "
              f"{(results_df['p_value'] < self.args.p_value_threshold).sum()}")

        self.results['differential'] = results_df
        return results_df

    def method_elasticnet_cox(self) -> pd.DataFrame:
        """
        Elastic Net regularized Cox regression with cross-validation.

        This method performs feature selection using L1/L2 regularization
        and identifies the most important genes for survival prediction.

        Returns:
            DataFrame with gene names and their coefficients
        """
        print("\n=== Running Elastic Net Cox Regression ===")

        # Prepare data
        X = self.data[self.gene_names].values
        y = Surv.from_dataframe('OS', 'OS.year', self.data)

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit Elastic Net Cox model with cross-validation
        print("Performing cross-validation to find optimal alpha...")
        alphas = np.logspace(-4, 1, 50)

        cv_scores = []
        for alpha in alphas:
            estimator = CoxnetSurvivalAnalysis(
                l1_ratio=0.5,  # Elastic net (mix of L1 and L2)
                alpha_min_ratio=alpha,
                fit_baseline_model=True,
                n_alphas=1
            )

            # Cross-validation
            kf = KFold(n_splits=self.args.cv_folds, shuffle=True,
                      random_state=self.args.random_seed)
            fold_scores = []

            for train_idx, val_idx in kf.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                try:
                    estimator.fit(X_train, y_train)
                    score = estimator.score(X_val, y_val)
                    fold_scores.append(score)
                except:
                    fold_scores.append(np.nan)

            cv_scores.append(np.nanmean(fold_scores))

        # Select best alpha
        best_alpha = alphas[np.nanargmax(cv_scores)]
        print(f"Best alpha: {best_alpha:.6f}, CV score: {np.nanmax(cv_scores):.4f}")

        # Fit final model
        final_model = CoxnetSurvivalAnalysis(
            l1_ratio=0.5,
            alpha_min_ratio=best_alpha,
            fit_baseline_model=True,
            n_alphas=1
        )
        final_model.fit(X_scaled, y)

        # Extract coefficients
        coefs = final_model.coef_

        # Create results dataframe
        results = []
        for i, gene in enumerate(self.gene_names):
            if coefs[i] != 0:
                results.append({
                    'gene': gene,
                    'coefficient': coefs[i],
                    'abs_coefficient': abs(coefs[i])
                })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('abs_coefficient', ascending=False)

        print(f"Selected {len(results_df)} genes with non-zero coefficients")

        self.results['elasticnet'] = results_df
        return results_df

    def get_consensus_genes(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Get consensus genes that appear in multiple methods.

        Args:
            top_n: Number of top genes to consider from each method

        Returns:
            DataFrame with genes ranked by how many methods selected them
        """
        if top_n is None:
            top_n = self.args.n_top_genes

        print(f"\n=== Finding Consensus Genes (top {top_n} from each method) ===")

        gene_counts = {}
        gene_methods = {}

        for method_name, results_df in self.results.items():
            if method_name == 'elasticnet':
                # For elastic net, use genes with non-zero coefficients
                top_genes = results_df.head(top_n)['gene'].tolist()
            else:
                # For other methods, use top N by p-value
                top_genes = results_df.head(top_n)['gene'].tolist()

            for gene in top_genes:
                gene_counts[gene] = gene_counts.get(gene, 0) + 1
                if gene not in gene_methods:
                    gene_methods[gene] = []
                gene_methods[gene].append(method_name)

        # Create consensus dataframe
        consensus = []
        for gene, count in gene_counts.items():
            consensus.append({
                'gene': gene,
                'n_methods': count,
                'methods': ', '.join(gene_methods[gene])
            })

        consensus_df = pd.DataFrame(consensus)
        consensus_df = consensus_df.sort_values('n_methods', ascending=False)

        print(f"\nGenes appearing in multiple methods:")
        for n in range(len(self.args.methods), 0, -1):
            count = (consensus_df['n_methods'] == n).sum()
            if count > 0:
                print(f"  {n} methods: {count} genes")

        return consensus_df

    def visualize_results(self, output_dir: str):
        """Generate visualization plots for the results"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        pdf_path = os.path.join(output_dir, 'marker_gene_analysis.pdf')

        with PdfPages(pdf_path) as pdf:
            # Plot 1: Volcano plots for each method
            if 'cox' in self.results:
                self._plot_volcano_cox(pdf)

            if 'differential' in self.results:
                self._plot_volcano_differential(pdf)

            # Plot 2: Top genes barplot
            self._plot_top_genes(pdf)

            # Plot 3: Kaplan-Meier curves for top genes
            self._plot_km_curves(pdf)

            # Plot 4: Heatmap of top genes
            self._plot_heatmap(pdf)

            # Plot 5: Consensus genes Venn-like plot
            self._plot_consensus(pdf)

        print(f"\nVisualizations saved to {pdf_path}")

    def _plot_volcano_cox(self, pdf):
        """Volcano plot for Cox regression results"""
        fig, ax = plt.subplots(figsize=(10, 8))

        df = self.results['cox']

        # Plot all points
        ax.scatter(df['coef'], df['log10_p'], alpha=0.3, s=10, color='gray')

        # Highlight significant genes
        sig_mask = df['p_value'] < self.args.p_value_threshold
        ax.scatter(df.loc[sig_mask, 'coef'],
                  df.loc[sig_mask, 'log10_p'],
                  alpha=0.6, s=20, color='red', label='Significant')

        # Add threshold line
        ax.axhline(-np.log10(self.args.p_value_threshold),
                  color='blue', linestyle='--', alpha=0.5)

        ax.set_xlabel('Cox Coefficient (log HR)', fontsize=12)
        ax.set_ylabel('-log10(p-value)', fontsize=12)
        ax.set_title('Volcano Plot: Cox Proportional Hazards', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    def _plot_volcano_differential(self, pdf):
        """Volcano plot for differential expression results"""
        fig, ax = plt.subplots(figsize=(10, 8))

        df = self.results['differential']

        # Plot all points
        ax.scatter(df['log_FC'], df['log10_p'], alpha=0.3, s=10, color='gray')

        # Highlight significant genes
        sig_mask = df['p_value'] < self.args.p_value_threshold
        ax.scatter(df.loc[sig_mask, 'log_FC'],
                  df.loc[sig_mask, 'log10_p'],
                  alpha=0.6, s=20, color='red', label='Significant')

        # Add threshold line
        ax.axhline(-np.log10(self.args.p_value_threshold),
                  color='blue', linestyle='--', alpha=0.5)

        ax.set_xlabel('Log Fold Change (Event vs Censored)', fontsize=12)
        ax.set_ylabel('-log10(p-value)', fontsize=12)
        ax.set_title('Volcano Plot: Differential Expression', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    def _plot_top_genes(self, pdf):
        """Bar plot of top genes from each method"""
        n_methods = len(self.results)
        fig, axes = plt.subplots(n_methods, 1, figsize=(12, 4*n_methods))

        if n_methods == 1:
            axes = [axes]

        for idx, (method_name, results_df) in enumerate(self.results.items()):
            ax = axes[idx]

            # Get top 20 genes
            top_genes = results_df.head(20)

            if method_name == 'elasticnet':
                y_vals = top_genes['abs_coefficient']
                ylabel = 'Absolute Coefficient'
            else:
                y_vals = top_genes['log10_p']
                ylabel = '-log10(p-value)'

            # Plot
            bars = ax.barh(range(len(top_genes)), y_vals)
            ax.set_yticks(range(len(top_genes)))
            ax.set_yticklabels(top_genes['gene'])
            ax.set_xlabel(ylabel, fontsize=11)
            ax.set_title(f'Top 20 Genes: {method_name.upper()}',
                        fontsize=12, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    def _plot_km_curves(self, pdf):
        """Kaplan-Meier curves for top consensus genes"""
        consensus = self.get_consensus_genes(top_n=self.args.n_top_genes)
        top_genes = consensus.head(9)['gene'].tolist()

        if len(top_genes) == 0:
            return

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()

        kmf_high = KaplanMeierFitter()
        kmf_low = KaplanMeierFitter()

        for idx, gene in enumerate(top_genes):
            ax = axes[idx]

            # Split by median
            median_expr = self.data[gene].median()
            high_expr = self.data[gene] > median_expr

            # Fit KM curves
            kmf_high.fit(
                self.data.loc[high_expr, 'OS.year'],
                self.data.loc[high_expr, 'OS'],
                label='High expression'
            )
            kmf_low.fit(
                self.data.loc[~high_expr, 'OS.year'],
                self.data.loc[~high_expr, 'OS'],
                label='Low expression'
            )

            # Plot
            kmf_high.plot_survival_function(ax=ax, ci_show=True)
            kmf_low.plot_survival_function(ax=ax, ci_show=True)

            # Log-rank test
            result = logrank_test(
                self.data.loc[high_expr, 'OS.year'],
                self.data.loc[~high_expr, 'OS.year'],
                self.data.loc[high_expr, 'OS'],
                self.data.loc[~high_expr, 'OS']
            )

            ax.set_title(f'{gene}\np = {result.p_value:.2e}', fontsize=10)
            ax.set_xlabel('Time (years)', fontsize=9)
            ax.set_ylabel('Survival probability', fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        # Remove empty subplots
        for idx in range(len(top_genes), 9):
            fig.delaxes(axes[idx])

        plt.suptitle('Kaplan-Meier Curves: Top Consensus Genes',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    def _plot_heatmap(self, pdf):
        """Heatmap of top genes expression"""
        consensus = self.get_consensus_genes(top_n=30)
        top_genes = consensus.head(30)['gene'].tolist()

        if len(top_genes) == 0:
            return

        # Prepare data
        expr_data = self.data[top_genes].T

        # Sort samples by survival time
        sample_order = self.data.sort_values('OS.year').index
        expr_data = expr_data[sample_order]

        # Create annotation for outcome
        outcome_colors = self.data.loc[sample_order, 'OS'].map({0: 'lightblue', 1: 'salmon'})

        # Plot
        fig, (ax_hm, ax_cb) = plt.subplots(2, 1, figsize=(14, 10),
                                           gridspec_kw={'height_ratios': [20, 1]})

        # Heatmap
        sns.heatmap(expr_data, cmap='RdBu_r', center=expr_data.values.mean(),
                   cbar_ax=ax_cb, cbar_kws={'orientation': 'horizontal'},
                   xticklabels=False, yticklabels=True, ax=ax_hm)

        ax_hm.set_xlabel('Samples (sorted by survival time)', fontsize=11)
        ax_hm.set_ylabel('Genes', fontsize=11)
        ax_hm.set_title('Expression Heatmap: Top 30 Consensus Genes',
                       fontsize=14, fontweight='bold')

        # Add outcome color bar
        for i, (idx, color) in enumerate(zip(sample_order, outcome_colors)):
            ax_hm.axvline(i, color=color, linewidth=0.5, alpha=0.5)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    def _plot_consensus(self, pdf):
        """Bar plot showing gene overlap between methods"""
        consensus = self.get_consensus_genes(top_n=self.args.n_top_genes)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Count genes by number of methods
        method_counts = consensus['n_methods'].value_counts().sort_index(ascending=False)

        bars = ax.bar(method_counts.index, method_counts.values,
                     color=sns.color_palette('viridis', len(method_counts)))

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_xlabel('Number of Methods', fontsize=12)
        ax.set_ylabel('Number of Genes', fontsize=12)
        ax.set_title('Consensus Genes Across Methods', fontsize=14, fontweight='bold')
        ax.set_xticks(method_counts.index)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    def save_results(self, output_dir: str):
        """Save all results to CSV files"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Save individual method results
        for method_name, results_df in self.results.items():
            output_path = os.path.join(output_dir, f'{method_name}_results.csv')
            results_df.to_csv(output_path, index=False)
            print(f"Saved {method_name} results to {output_path}")

        # Save consensus genes
        consensus = self.get_consensus_genes()
        consensus_path = os.path.join(output_dir, 'consensus_genes.csv')
        consensus.to_csv(consensus_path, index=False)
        print(f"Saved consensus genes to {consensus_path}")

        # Save top genes from each method combined
        top_genes_combined = []
        for method_name, results_df in self.results.items():
            top_n = results_df.head(self.args.n_top_genes).copy()
            top_n['method'] = method_name
            top_genes_combined.append(top_n)

        combined_path = os.path.join(output_dir, 'top_genes_all_methods.csv')
        pd.concat(top_genes_combined).to_csv(combined_path, index=False)
        print(f"Saved combined top genes to {combined_path}")


def main():
    """Main execution function"""
    # Parse arguments
    args = Args().parse_args()

    print("="*80)
    print("MARKER GENE SELECTION FOR PROGNOSIS PREDICTION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Input file: {args.input_file}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Methods: {', '.join(args.methods)}")
    print(f"  Top N genes: {args.n_top_genes}")
    print(f"  P-value threshold: {args.p_value_threshold}")
    print(f"  CV folds: {args.cv_folds}")
    print(f"  Random seed: {args.random_seed}")

    # Initialize selector
    selector = MarkerGeneSelector(args)

    # Load data
    selector.load_data()

    # Run selected methods
    if 'cox' in args.methods:
        selector.method_cox_regression()

    if 'logrank' in args.methods:
        selector.method_logrank_test()

    if 'differential' in args.methods:
        selector.method_differential_expression()

    if 'elasticnet' in args.methods:
        selector.method_elasticnet_cox()

    # Get consensus genes
    consensus = selector.get_consensus_genes()
    print("\n=== Top 20 Consensus Genes ===")
    print(consensus.head(20).to_string(index=False))

    # Save results
    selector.save_results(args.output_dir)

    # Generate visualizations
    if args.visualization:
        print("\n=== Generating Visualizations ===")
        selector.visualize_results(args.output_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}/")
    print(f"  - Individual method results: {{method}}_results.csv")
    print(f"  - Consensus genes: consensus_genes.csv")
    print(f"  - Combined top genes: top_genes_all_methods.csv")
    if args.visualization:
        print(f"  - Visualizations: marker_gene_analysis.pdf")


if __name__ == "__main__":
    main()
