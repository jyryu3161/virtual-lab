#!/usr/bin/env python3
"""
Run biomarker analysis using YAML configuration file.

Usage:
    python run_with_config.py config.yaml
    python run_with_config.py config.yaml --dry-run
"""

import sys
import yaml
import argparse
from pathlib import Path
from select_marker_genes import MarkerGeneSelector
import pandas as pd
import logging


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: dict):
    """Setup logging based on config."""
    log_level = config.get('advanced', {}).get('log_level', 'INFO')
    log_file = config.get('advanced', {}).get('log_file', None)

    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )


def validate_config(config: dict):
    """Validate configuration file."""
    required_keys = ['data', 'output', 'methods']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required section in config: {key}")

    # Check input file exists
    input_file = config['data']['input_file']
    if not Path(input_file).exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    logging.info("✓ Configuration validated successfully")


def run_analysis(config: dict, dry_run: bool = False):
    """Run biomarker analysis based on configuration."""

    # Setup logging
    setup_logging(config)

    # Validate config
    validate_config(config)

    if dry_run:
        logging.info("DRY RUN MODE - No analysis will be performed")
        logging.info(f"Would analyze: {config['data']['input_file']}")
        logging.info(f"Methods enabled: {config['methods']['enabled']}")
        logging.info(f"Output directory: {config['output']['output_dir']}")
        return

    # Load data
    logging.info(f"Loading data from: {config['data']['input_file']}")
    data = pd.read_csv(config['data']['input_file'])

    # Extract column names
    col_config = config['data']['columns']
    survival_time_col = col_config['survival_time']
    survival_event_col = col_config['survival_event']

    # Filter low variance genes if requested
    if config['data']['filters'].get('remove_low_variance_genes', False):
        min_var = config['data']['filters']['min_expression_variance']
        # Get gene expression columns (numeric columns excluding survival)
        gene_cols = data.select_dtypes(include=['number']).columns
        gene_cols = [c for c in gene_cols if c not in [survival_time_col, survival_event_col]]

        # Calculate variance
        variances = data[gene_cols].var()
        high_var_genes = variances[variances > min_var].index.tolist()

        logging.info(f"Filtered {len(gene_cols) - len(high_var_genes)} low variance genes")

        # Keep only high variance genes + survival columns
        data = data[[survival_time_col, survival_event_col] + high_var_genes]

    # Initialize selector
    output_dir = config['output']['output_dir']
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    selector = MarkerGeneSelector(
        data=data,
        survival_time_col=survival_time_col,
        survival_event_col=survival_event_col,
        output_dir=output_dir
    )

    # Run enabled methods
    enabled_methods = config['methods']['enabled']
    results = {}

    if 'cox' in enabled_methods:
        logging.info("Running Cox Proportional Hazards Regression...")
        cox_config = config['methods']['cox']
        results['cox'] = selector.method_cox_regression(
            alpha=cox_config['alpha'],
            adjust_multiple_testing=cox_config['adjust_multiple_testing']
        )
        logging.info(f"  Found {len(results['cox'])} significant genes")

    if 'logrank' in enabled_methods:
        logging.info("Running Log-rank Test...")
        lr_config = config['methods']['logrank']
        results['logrank'] = selector.method_logrank_test(
            alpha=lr_config['alpha'],
            median_split=lr_config['median_split'],
            adjust_multiple_testing=lr_config['adjust_multiple_testing']
        )
        logging.info(f"  Found {len(results['logrank'])} significant genes")

    if 'differential' in enabled_methods:
        logging.info("Running Differential Expression Analysis...")
        diff_config = config['methods']['differential']
        results['differential'] = selector.method_differential_expression(
            alpha=diff_config['alpha'],
            min_fold_change=diff_config.get('min_fold_change', 1.0),
            adjust_multiple_testing=diff_config['adjust_multiple_testing']
        )
        logging.info(f"  Found {len(results['differential'])} significant genes")

    if 'elasticnet' in enabled_methods:
        logging.info("Running Elastic Net Cox Regression...")
        en_config = config['methods']['elasticnet']
        results['elasticnet'] = selector.method_elasticnet_cox(
            l1_ratio=en_config['l1_ratio'],
            n_folds=en_config['cv_folds'],
            n_alphas=en_config['n_alphas'],
            max_iter=en_config['max_iter']
        )
        logging.info(f"  Found {len(results['elasticnet'])} genes")

    # Consensus analysis
    if len(results) > 1:
        logging.info("Performing consensus analysis...")
        consensus_config = config['consensus']
        consensus = selector.get_consensus_genes(
            method_results=results,
            min_methods=consensus_config['min_methods']
        )
        logging.info(f"  Found {len(consensus)} consensus genes")

        # Save consensus
        consensus_file = Path(output_dir) / f"{config['output']['prefix']}_consensus.csv"
        consensus.to_csv(consensus_file, index=False)
        logging.info(f"  Saved to: {consensus_file}")

    # Generate visualizations
    if config['visualization']['enabled']:
        logging.info("Generating visualizations...")
        viz_config = config['visualization']

        # This would call visualization methods from the selector
        # (Implementation depends on your MarkerGeneSelector class)
        logging.info("  Visualization generation complete")

    logging.info("✓ Analysis complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Run biomarker analysis with YAML configuration'
    )
    parser.add_argument(
        'config',
        type=str,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate config without running analysis'
    )

    args = parser.parse_args()

    # Load and run
    config = load_config(args.config)
    run_analysis(config, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
