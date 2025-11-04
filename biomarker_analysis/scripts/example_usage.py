"""
Example: Simple Biomarker Analysis

This script demonstrates a simple biomarker analysis workflow
using the MarkerGeneSelector class.
"""

from select_marker_genes import MarkerGeneSelector, Args

def main():
    """Run a simple biomarker analysis example"""

    print("="*80)
    print("SIMPLE BIOMARKER ANALYSIS EXAMPLE")
    print("="*80)
    print()

    # Configure analysis
    args = Args(
        input_file="../../Example_TCGA_TNBC_data.csv",
        output_dir="../../example_biomarker_results",
        n_top_genes=20,
        p_value_threshold=0.05,
        methods=["cox", "logrank"],  # Using only 2 fast methods for demo
        visualization=True,
        random_seed=42
    )

    print("Configuration:")
    print(f"  Input: {args.input_file}")
    print(f"  Output: {args.output_dir}")
    print(f"  Methods: {', '.join(args.methods)}")
    print(f"  Top N genes: {args.n_top_genes}")
    print()

    # Initialize selector
    selector = MarkerGeneSelector(args)

    # Load data
    print("Step 1: Loading data...")
    data = selector.load_data()
    print(f"  ✓ Loaded {len(data)} samples with {len(selector.gene_names)} genes")
    print()

    # Run Cox regression
    print("Step 2: Running Cox Proportional Hazards analysis...")
    cox_results = selector.method_cox_regression()
    print(f"  ✓ Analyzed {len(cox_results)} genes")
    print(f"  ✓ Found {(cox_results['p_value'] < 0.05).sum()} significant genes (p < 0.05)")
    print()

    # Run Log-rank test
    print("Step 3: Running Log-rank test...")
    logrank_results = selector.method_logrank_test()
    print(f"  ✓ Analyzed {len(logrank_results)} genes")
    print(f"  ✓ Found {(logrank_results['p_value'] < 0.05).sum()} significant genes (p < 0.05)")
    print()

    # Get consensus genes
    print("Step 4: Finding consensus genes...")
    consensus = selector.get_consensus_genes()
    print(f"  ✓ Found {len(consensus)} genes appearing in at least one method")
    print()

    # Display top genes
    print("="*80)
    print("TOP 10 CONSENSUS GENES")
    print("="*80)
    print()
    print(consensus.head(10).to_string(index=False))
    print()

    # Save results
    print("="*80)
    print("Step 5: Saving results...")
    selector.save_results(args.output_dir)
    print(f"  ✓ Results saved to {args.output_dir}/")
    print()

    # Generate visualizations
    print("Step 6: Generating visualizations...")
    selector.visualize_results(args.output_dir)
    print(f"  ✓ Visualizations saved to {args.output_dir}/marker_gene_analysis.pdf")
    print()

    print("="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print()
    print("Next steps:")
    print("  1. Review results in:", args.output_dir)
    print("  2. Examine visualizations in: marker_gene_analysis.pdf")
    print("  3. Validate top genes in independent datasets")
    print()


if __name__ == "__main__":
    main()
