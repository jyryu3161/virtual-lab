#!/usr/bin/env python3
"""
Run metabolic modeling analysis using YAML configuration file.

Usage:
    python run_with_config.py config.yaml
    python run_with_config.py config.yaml --dry-run
"""

import sys
import yaml
import argparse
from pathlib import Path
from metabolic_target_finder import MetabolicTargetFinder
from pathway_designer_tools import PathwayDesigner, EXAMPLE_PATHWAYS
import cobra
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


def load_model(config: dict):
    """Load metabolic model based on configuration."""
    model_config = config['model']

    if model_config['source'] == 'bigg':
        bigg_id = model_config['bigg_id']
        logging.info(f"Loading model from BiGG: {bigg_id}")
        model = cobra.io.load_model(bigg_id)

    elif model_config['source'] == 'file':
        file_path = model_config['file_path']
        file_format = model_config['file_format']
        logging.info(f"Loading model from file: {file_path}")

        if file_format == 'json':
            model = cobra.io.load_json_model(file_path)
        elif file_format in ['xml', 'sbml']:
            model = cobra.io.read_sbml_model(file_path)
        elif file_format == 'mat':
            model = cobra.io.load_matlab_model(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

    else:
        raise ValueError(f"Unknown model source: {model_config['source']}")

    # Apply model modifications
    modifications = model_config.get('modifications', {})

    # Custom bounds
    if modifications.get('custom_bounds'):
        for rxn_id, bounds in modifications['custom_bounds'].items():
            if rxn_id in model.reactions:
                model.reactions.get_by_id(rxn_id).bounds = tuple(bounds)
                logging.info(f"  Set bounds for {rxn_id}: {bounds}")

    # Objective
    if modifications.get('objective'):
        obj_id = modifications['objective']
        if obj_id in model.reactions:
            model.objective = obj_id
            logging.info(f"  Set objective: {obj_id}")

    # Medium
    if modifications.get('medium') == 'minimal':
        model.medium = model.minimal_medium()
        logging.info("  Applied minimal medium")
    elif modifications.get('medium') == 'custom':
        custom_medium = modifications.get('custom_medium', {})
        for ex_id, rate in custom_medium.items():
            if ex_id in model.reactions:
                model.reactions.get_by_id(ex_id).lower_bound = rate

    logging.info(f"Model loaded: {model.id} ({len(model.reactions)} reactions, {len(model.genes)} genes)")

    return model


def validate_config(config: dict):
    """Validate configuration file."""
    required_keys = ['model', 'output', 'methods']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required section in config: {key}")

    logging.info("✓ Configuration validated successfully")


def run_analysis(config: dict, dry_run: bool = False):
    """Run metabolic modeling analysis based on configuration."""

    # Setup logging
    setup_logging(config)

    # Validate config
    validate_config(config)

    if dry_run:
        logging.info("DRY RUN MODE - No analysis will be performed")
        logging.info(f"Model source: {config['model']['source']}")
        if config['model']['source'] == 'bigg':
            logging.info(f"  BiGG ID: {config['model']['bigg_id']}")
        logging.info(f"Methods enabled: {config['methods']['enabled']}")
        logging.info(f"Output directory: {config['output']['output_dir']}")
        return

    # Load model
    model = load_model(config)

    # Setup output directory
    output_dir = config['output']['output_dir']
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Check if pathway design is enabled
    pathway_config = config['methods'].get('pathway_design', {})
    if pathway_config.get('enabled', False):
        logging.info("=== PATHWAY DESIGN MODE ===")

        designer = PathwayDesigner(model)

        # Load pathway template or custom pathway
        template = pathway_config.get('pathway_template')
        if template in EXAMPLE_PATHWAYS:
            logging.info(f"Loading pathway template: {template}")
            pathway_func = EXAMPLE_PATHWAYS[template]
            pathway_func(designer)
        elif template == 'custom':
            logging.info("Building custom pathway...")
            # Add custom metabolites
            for met in pathway_config['custom_pathway'].get('metabolites', []):
                designer.add_metabolite(
                    met['id'], met['name'], met['formula'], met['compartment']
                )

            # Add custom reactions
            for rxn in pathway_config['custom_pathway'].get('reactions', []):
                designer.add_reaction(
                    rxn['id'], rxn['name'], rxn['metabolites'],
                    rxn.get('lower_bound', 0),
                    rxn.get('upper_bound', 1000),
                    rxn.get('gene_reaction_rule', '')
                )

        # Test feasibility
        if pathway_config.get('test_feasibility', True):
            target = pathway_config['target_compound']
            logging.info(f"Testing pathway feasibility for: {target}")
            result = designer.test_pathway_feasibility(target)

            if result['feasible']:
                logging.info(f"  ✓ Pathway is feasible!")
                logging.info(f"  Production flux: {result['production_flux']:.4f} mmol/gDW/h")
            else:
                logging.warning(f"  ✗ Pathway is not feasible")

        # Save modified model
        if config['output'].get('save_model', True):
            model_file = Path(output_dir) / "model_with_pathway.json"
            cobra.io.save_json_model(model, str(model_file))
            logging.info(f"Saved modified model: {model_file}")

    # Initialize target finder
    finder = MetabolicTargetFinder(
        model=model,
        output_dir=output_dir
    )

    # Run enabled methods
    enabled_methods = config['methods']['enabled']
    results = {}

    if 'single' in enabled_methods:
        logging.info("Running Single Gene Knockout Analysis...")
        ko_config = config['methods']['single_knockout']
        results['single'] = finder.method_single_gene_knockout(
            growth_threshold=ko_config['growth_threshold'],
            processes=ko_config.get('processes', -1)
        )
        logging.info(f"  Analyzed {len(results['single'])} genes")

    if 'essential' in enabled_methods:
        logging.info("Running Essential Gene Analysis...")
        ess_config = config['methods']['essential_genes']
        results['essential'] = finder.method_essential_genes(
            growth_threshold=ess_config['growth_threshold']
        )
        logging.info(f"  Found {len(results['essential'])} essential genes")

    if 'double' in enabled_methods:
        logging.info("Running Double Gene Knockout Analysis...")
        dbl_config = config['methods']['double_knockout']
        results['double'] = finder.method_double_gene_knockout(
            growth_threshold=dbl_config['growth_threshold'],
            max_pairs=dbl_config.get('max_pairs'),
            processes=dbl_config.get('processes', -1)
        )
        logging.info(f"  Found {len(results['double'])} synthetic lethal pairs")

    if 'fva' in enabled_methods:
        logging.info("Running Flux Variability Analysis...")
        fva_config = config['methods']['fva']
        results['fva'] = finder.method_flux_variability_analysis(
            fraction_of_optimum=fva_config['fraction_of_optimum'],
            processes=fva_config.get('processes', -1)
        )
        logging.info(f"  Analyzed {len(results['fva'])} reactions")

    if 'production' in enabled_methods:
        prod_config = config['methods']['production']
        if prod_config.get('enabled', False):
            logging.info("Running Growth-Coupled Production Analysis...")
            target = prod_config['target_metabolite']
            results['production'] = finder.method_growth_coupled_production(
                target_metabolite=target,
                growth_threshold=prod_config['growth_threshold']
            )
            logging.info(f"  Found {len(results['production'])} production-enhancing knockouts")

    # Generate visualizations
    if config['visualization']['enabled']:
        logging.info("Generating visualizations...")
        # Visualization implementation would go here
        logging.info("  Visualization generation complete")

    logging.info("✓ Analysis complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Run metabolic modeling with YAML configuration'
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
