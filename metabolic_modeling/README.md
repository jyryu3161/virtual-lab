# Genome-Scale Metabolic Modeling

This module provides comprehensive tools for identifying gene knockout/knockdown targets using genome-scale metabolic models (GEMs) and COBRApy.

## Overview

Genome-scale metabolic modeling uses constraint-based analysis to predict cellular behavior and identify metabolic engineering targets. This implementation provides:

- **Single & double gene knockout simulations**
- **Essential gene identification**
- **Synthetic lethality discovery**
- **Flux variability analysis (FVA)**
- **Growth-coupled production analysis**
- **Comprehensive visualization**

## Directory Structure

```
metabolic_modeling/
├── scripts/
│   └── metabolic_target_finder.py    # Main analysis script
├── models/                            # Store your metabolic models here
├── examples/                          # Example models and data
├── tutorial_metabolic_modeling.ipynb  # Interactive tutorial
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Quick Start

### Method 1: Using YAML Configuration (Recommended) 🆕

```bash
cd metabolic_modeling

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
- ✅ All parameters in one place (model, methods, output)
- ✅ Easy to reproduce experiments
- ✅ Version control your analysis settings
- ✅ Share configurations with collaborators
- ✅ Support for pathway design configuration

See [Configuration Guide](#configuration-guide) for detailed parameter descriptions.

### Method 2: Using Pixi

```bash
# From the virtual-lab root directory

# Run complete analysis with E. coli core model
pixi run metabolic-analysis

# Launch interactive tutorial
pixi run metabolic-tutorial

# Quick test
pixi run test-metabolic
```

### Method 3: Command Line

```bash
cd metabolic_modeling/scripts

# Use E. coli core model (fast, for testing)
python metabolic_target_finder.py \
    --model_id textbook \
    --output_dir ../../metabolic_results \
    --ko_methods single essential fva \
    --visualization

# Use larger E. coli model (iML1515)
python metabolic_target_finder.py \
    --model_id iML1515 \
    --output_dir ../../metabolic_results \
    --ko_methods single essential fva \
    --growth_threshold 0.1 \
    --visualization

# Use custom SBML model file
python metabolic_target_finder.py \
    --model_file path/to/model.xml \
    --output_dir ../../metabolic_results \
    --ko_methods single double essential \
    --visualization
```

### Method 3: Jupyter Notebook

```bash
jupyter notebook tutorial_metabolic_modeling.ipynb
```

### Method 4: Python API

```python
from metabolic_target_finder import MetabolicTargetFinder, Args

# Configure analysis
args = Args(
    model_id="iML1515",  # or model_file="path/to/model.xml"
    output_dir="results",
    ko_methods=["single", "essential", "fva"],
    growth_threshold=0.1
)

# Run analysis
finder = MetabolicTargetFinder(args)
finder.load_model()
finder.method_single_gene_knockout()
finder.method_essential_genes()
finder.method_flux_variability_analysis()

# Save and visualize
finder.save_results(args.output_dir)
finder.visualize_results(args.output_dir)
```

## Supported Model Formats

### BiGG Models Database

Load pre-built models directly from BiGG:

```python
import cobra

# E. coli
model = cobra.io.load_model("iML1515")  # 2,712 reactions, 1,877 genes
model = cobra.io.load_model("textbook")  # Core model (fast)

# Human
model = cobra.io.load_model("Recon3D")  # 13,543 reactions, 3,288 genes

# Yeast
model = cobra.io.load_model("iMM904")  # 1,577 reactions, 904 genes
```

### File Formats

- **SBML** (`.xml`, `.sbml`): Systems Biology Markup Language
- **JSON** (`.json`): COBRApy JSON format
- **MAT** (`.mat`): MATLAB format

```python
# Load from file
model = cobra.io.read_sbml_model("path/to/model.xml")
model = cobra.io.load_json_model("path/to/model.json")
model = cobra.io.load_matlab_model("path/to/model.mat")
```

## Analysis Methods

### 1. Single Gene Knockout

Systematically delete each gene and measure growth impact.

```bash
python metabolic_target_finder.py \
    --model_id textbook \
    --ko_methods single \
    --growth_threshold 0.1
```

**Output:**
- `single_knockout_results.csv`: All genes with growth effects
- Essential vs. non-essential classification
- Growth reduction percentages

**Use cases:**
- Identify drug targets (essential genes)
- Find metabolic engineering targets (non-essential growth-reducing)
- Understand gene importance

### 2. Essential Gene Identification

Genes whose deletion causes lethality (growth < threshold).

```bash
python metabolic_target_finder.py \
    --model_id iML1515 \
    --ko_methods essential \
    --growth_threshold 0.1
```

**Output:**
- `essential_genes_results.csv`: Essential genes with annotations
- Reaction involvement
- Gene criticality ranking

**Use cases:**
- Antibiotic target discovery
- Cancer drug targets
- Understanding core metabolism

### 3. Double Gene Knockout (Synthetic Lethality)

Find gene pairs that are non-essential individually but lethal together.

```bash
python metabolic_target_finder.py \
    --model_id iML1515 \
    --ko_methods double \
    --growth_threshold 0.1
```

**Output:**
- `double_knockout_results.csv`: All gene pair combinations
- Synthetic lethal pairs
- Synergy scores

**Use cases:**
- Combination therapy design
- Robustness analysis
- Multi-target drugs

**Note:** Computationally intensive! ~O(n²) where n = number of genes.

### 4. Flux Variability Analysis (FVA)

Identify min/max flux ranges for each reaction at near-optimal growth.

```bash
python metabolic_target_finder.py \
    --model_id iML1515 \
    --ko_methods fva \
    --fva_fraction 0.95
```

**Output:**
- `fva_results.csv`: Flux ranges for all reactions
- Flexible vs. rigid reactions
- Essential flux pathways

**Use cases:**
- Identify metabolic bottlenecks
- Find robust engineering targets
- Understand metabolic flexibility

### 5. Growth-Coupled Production

Find knockouts that couple product formation with growth.

```bash
python metabolic_target_finder.py \
    --model_id iML1515 \
    --ko_methods production \
    --target_metabolite succ_c
```

**Output:**
- `growth_coupled_results.csv`: Knockouts ranked by production/growth ratio
- Production rates
- Growth impacts

**Use cases:**
- Metabolic engineering for bioproduction
- Optimize yield
- Evolution-based strain improvement

### 6. Flux Sampling

Sample solution space to explore alternative metabolic states.

```bash
python metabolic_target_finder.py \
    --model_id iML1515 \
    --ko_methods sampling \
    --n_samples 1000
```

**Output:**
- `flux_sampling_results.csv`: Flux statistics across samples
- High-variability reactions
- Metabolic state distributions

**Use cases:**
- Explore alternative pathways
- Understand metabolic uncertainty
- Identify regulatory targets

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_file` | str | None | Path to model file (SBML/JSON/MAT) |
| `--model_id` | str | `iML1515` | BiGG model ID if no file provided |
| `--output_dir` | str | `metabolic_results` | Output directory |
| `--objective` | str | None | Objective reaction (default: biomass) |
| `--target_metabolite` | str | None | Target for production analysis |
| `--ko_methods` | list | `[single, essential, fva]` | Methods to run |
| `--growth_threshold` | float | `0.1` | Essential gene threshold (fraction) |
| `--n_samples` | int | `1000` | Number of flux samples |
| `--fva_fraction` | float | `0.95` | Fraction of optimal for FVA |
| `--top_n_targets` | int | `50` | Number of top targets to report |
| `--visualization` | bool | `True` | Generate plots |
| `--random_seed` | int | `42` | Random seed |

## Available Methods

Use `--ko_methods` to select analyses:

- `single`: Single gene knockout
- `double`: Double gene knockout (slow!)
- `essential`: Essential gene identification
- `synthetic_lethality`: Find synthetic lethal pairs
- `fva`: Flux variability analysis
- `sampling`: Flux sampling
- `production`: Growth-coupled production (requires `--target_metabolite`)

## Output Files

### Results Files

- **`model_summary.csv`**: Model statistics and information
- **`single_knockout_results.csv`**: All single KO results
  - Columns: `gene_id`, `growth_rate`, `growth_fraction`, `growth_reduction`, `status`
- **`essential_genes_results.csv`**: Essential genes with details
  - Columns: `gene_id`, `gene_name`, `growth_fraction`, `n_reactions`, `reaction_ids`
- **`double_knockout_results.csv`**: Double KO results
  - Columns: `gene1`, `gene2`, `double_ko_fraction`, `synergy_score`, `synthetic_lethal`
- **`fva_results.csv`**: Flux ranges for all reactions
  - Columns: `reaction_id`, `minimum`, `maximum`, `flux_span`, `subsystem`
- **`growth_coupled_results.csv`**: Production-enhancing KOs
  - Columns: `gene_id`, `ko_growth`, `production`, `production_per_growth`
- **`flux_sampling_results.csv`**: Flux statistics
  - Columns: `reaction_id`, `mean_flux`, `std_flux`, `min_flux`, `max_flux`, `cv`

### Visualization File

- **`metabolic_analysis.pdf`**: Multi-page PDF with:
  - Knockout growth distribution
  - Essential genes classification
  - Synthetic lethality network
  - FVA flux ranges
  - Growth-coupled production plots

## Metabolic Models

### Recommended Models

**For Learning/Testing:**
- `textbook`: E. coli core model (95 reactions, 137 metabolites, 137 genes)
  - Fast analysis (~1 minute)
  - Good for tutorials

**For Production Use:**

**E. coli:**
- `iML1515`: Latest E. coli model (2,712 reactions, 1,877 genes)
  - Most comprehensive E. coli model
  - ~10-30 minutes for single KO

**Human:**
- `Recon3D`: Human metabolism (13,543 reactions, 3,288 genes)
  - Most comprehensive human model
  - Hours for complete analysis
  - Use for drug target discovery

**Yeast:**
- `iMM904`: S. cerevisiae (1,577 reactions, 904 genes)
  - Well-validated yeast model
  - Biotechnology applications

### Custom Models

To use your own model:

1. Ensure it's in SBML, JSON, or MAT format
2. Place in `metabolic_modeling/models/` directory
3. Use `--model_file` flag:

```bash
python metabolic_target_finder.py \
    --model_file ../models/my_model.xml \
    --output_dir ../results
```

## Performance Considerations

### Expected Runtime

| Model Size | Method | Approximate Time |
|------------|--------|------------------|
| Core (137 genes) | Single KO | ~30 seconds |
| Core (137 genes) | FVA | ~1-2 minutes |
| Core (137 genes) | Double KO | ~5 minutes |
| iML1515 (1,877 genes) | Single KO | ~10-30 minutes |
| iML1515 (1,877 genes) | FVA | ~30-60 minutes |
| iML1515 (1,877 genes) | Double KO | Hours to days |
| Recon3D (3,288 genes) | Single KO | ~1-2 hours |
| Recon3D (3,288 genes) | FVA | ~2-4 hours |

### Memory Requirements

- Small models (<500 reactions): ~500 MB
- Medium models (1,000-3,000 reactions): ~1-2 GB
- Large models (>5,000 reactions): ~3-5 GB

### Optimization Tips

1. **Start with core model** for testing
2. **Use faster solver** (CPLEX or Gurobi vs. GLPK)
3. **Limit double KO** to candidate genes only
4. **Reduce FVA fraction** (0.90 instead of 0.99)
5. **Parallelize** on HPC clusters

## Installing Optimization Solvers

COBRApy requires an optimization solver. Options:

### GLPK (Free, Default)

```bash
# Conda (recommended)
conda install -c conda-forge glpk

# Or pip
pip install swiglpk
```

### CPLEX (Commercial, Fast)

Free for academics: https://www.ibm.com/academic/technology/data-science

```bash
pip install cplex
```

### Gurobi (Commercial, Fast)

Free for academics: https://www.gurobi.com/academia/

```bash
pip install gurobipy
```

## Troubleshooting

### Common Issues

**1. No solver available**
```
OptimizationError: Solver status is 'failed'
```
**Solution**: Install a solver (GLPK, CPLEX, or Gurobi)

**2. Model loading fails**
```
FileNotFoundError: Model not found
```
**Solution**: Check model_id spelling or file path

**3. Memory error**
```
MemoryError: Unable to allocate array
```
**Solution**: Use smaller model or increase RAM

**4. Slow analysis**
```
Analysis taking too long...
```
**Solution**:
- Use faster solver (CPLEX/Gurobi)
- Reduce number of methods
- Use smaller model for testing

**5. Import errors**
```
ImportError: No module named 'cobra'
```
**Solution**: Install requirements
```bash
pip install -r requirements.txt
# Or
pixi install
```

## Example Use Cases

### Use Case 1: Antibiotic Target Discovery

Find essential genes in pathogenic bacteria:

```bash
python metabolic_target_finder.py \
    --model_file models/pathogen_model.xml \
    --ko_methods essential \
    --growth_threshold 0.05 \
    --output_dir antibiotic_targets
```

### Use Case 2: Biofuel Production

Optimize E. coli for ethanol production:

```bash
python metabolic_target_finder.py \
    --model_id iML1515 \
    --ko_methods single production \
    --target_metabolite etoh_c \
    --output_dir biofuel_engineering
```

### Use Case 3: Cancer Drug Targets

Find synthetic lethal pairs in cancer metabolism:

```bash
python metabolic_target_finder.py \
    --model_id Recon3D \
    --ko_methods double essential \
    --growth_threshold 0.1 \
    --output_dir cancer_targets
```

### Use Case 4: Metabolic Engineering

Design strain for succinate production:

```bash
python metabolic_target_finder.py \
    --model_id iML1515 \
    --ko_methods single fva production \
    --target_metabolite succ_c \
    --output_dir succinate_production
```

## Resources

### Documentation

- **COBRApy**: https://cobrapy.readthedocs.io/
- **BiGG Models**: http://bigg.ucsd.edu/
- **SBML**: http://sbml.org/
- **Virtual Lab**: See main README.md

### Papers

1. **Constraint-based modeling**:
   - Orth et al. (2010). "What is flux balance analysis?" *Nature Biotechnology*.

2. **COBRApy**:
   - Ebrahim et al. (2013). "COBRApy: COnstraints-Based Reconstruction and Analysis for Python." *BMC Systems Biology*.

3. **E. coli iML1515**:
   - Monk et al. (2017). "iML1515, a knowledgebase that computes Escherichia coli traits." *Nature Biotechnology*.

4. **Human Recon3D**:
   - Brunk et al. (2018). "Recon3D enables a three-dimensional view of gene variation in human metabolism." *Nature Biotechnology*.

### Tutorials

- **COBRApy getting started**: https://cobrapy.readthedocs.io/en/latest/getting_started.html
- **Escher** (flux visualization): https://escher.github.io/
- **MEMOTE** (model quality control): https://memote.io/

## Methodology: How It Works

### Overall Approach

Genome-scale metabolic modeling uses **constraint-based analysis** to predict cellular behavior without requiring kinetic parameters:

```
Metabolic Network → Constraints → Optimization → Flux Predictions → Phenotype Predictions
```

#### Step 1: Model Representation

**Stoichiometric Matrix (S)**:
```
S·v = 0  (steady-state assumption)

where:
  S = m × n matrix (m metabolites, n reactions)
  v = flux vector (reaction rates)
  0 = metabolite accumulation (steady state)
```

**Example**:
```
Reaction: Glucose + ATP → G6P + ADP
Stoichiometry: 1 glc + 1 atp → 1 g6p + 1 adp

Matrix row for glucose: [-1, 0, 0, ...]  (consumed)
Matrix row for g6p:     [+1, 0, 0, ...]  (produced)
```

#### Step 2: Constraints

Three types of constraints define the solution space:

**1. Stoichiometric constraints**: `S·v = 0`

**2. Flux bounds** (thermodynamics, capacity):
```
vₘᵢₙ ≤ v ≤ vₘₐₓ

Examples:
  Irreversible reaction: 0 ≤ v ≤ 1000
  Reversible reaction: -1000 ≤ v ≤ 1000
  Glucose uptake: -10 ≤ v_glc ≤ 0  (limited to 10 mmol/gDW/h)
```

**3. Objective function** (typically biomass maximization):
```
maximize Z = c^T · v

where:
  c = objective coefficient vector
  (usually c_biomass = 1, all others = 0)
```

#### Step 3: Flux Balance Analysis (FBA)

**Mathematical formulation**:
```
maximize    c^T · v
subject to  S · v = 0
            lb ≤ v ≤ ub
```

**Solution**:
- Linear programming problem
- Finds optimal flux distribution
- Predicts growth rate and metabolite production

**Assumptions**:
- Steady state (no metabolite accumulation)
- Optimal behavior (cells maximize objective)
- No enzyme kinetics required
- No regulatory information needed

### Detailed Method Descriptions

#### 1. Single Gene Knockout Analysis

**Algorithm**:
```python
wild_type_growth = FBA(model)

results = []
for gene in model.genes:
    with model:  # Temporary context
        gene.knock_out()  # Delete gene

        # Update reaction bounds based on GPR
        # (Gene-Protein-Reaction associations)
        update_reaction_bounds()

        mutant_growth = FBA(model)
        growth_fraction = mutant_growth / wild_type_growth

        results.append({
            'gene': gene.id,
            'growth_fraction': growth_fraction,
            'essential': growth_fraction < threshold
        })

return results
```

**Gene-Protein-Reaction (GPR) Rules**:
```
Gene A → Enzyme A → Reaction 1

GPR logic:
  - AND: "geneA and geneB" (both required, complex)
  - OR:  "geneA or geneB" (either sufficient, isozymes)

Example:
  Reaction: "gene123 or (gene456 and gene789)"

  Knockout gene123: Reaction still active (gene456+789 work)
  Knockout gene456: Reaction inactive (need both 456 AND 789)
```

**Classification**:
- **Essential**: `growth < 0.05 × wild_type` → lethal deletion
- **Important**: `0.05 < growth < 0.5` → severely impaired
- **Minor**: `0.5 < growth < 0.95` → slightly impaired
- **Neutral**: `growth ≥ 0.95` → no significant effect

#### 2. Flux Variability Analysis (FVA)

**Purpose**: Find the range of fluxes each reaction can carry while maintaining near-optimal growth.

**Algorithm**:
```python
optimal_growth = FBA(model)
min_growth = fraction × optimal_growth  # e.g., 0.9 × optimal

for reaction in model.reactions:
    # Minimize flux through this reaction
    model.objective = reaction
    min_flux = FBA(model, sense='minimize')

    # Maximize flux through this reaction
    max_flux = FBA(model, sense='maximize')

    flux_range = max_flux - min_flux

    # Classify reaction
    if min_flux == 0 and max_flux == 0:
        status = "blocked"
    elif min_flux * max_flux > 0:
        status = "essential" (always active)
    else:
        status = "variable"
```

**Interpretation**:
- **Narrow range**: Tightly constrained, less flexibility
- **Wide range**: High flexibility, redundant pathways
- **Zero range (blocked)**: Dead-end reaction, can be removed
- **Essential flux**: Required for optimal growth

**Applications**:
- Identify engineering targets (variable but important)
- Find metabolic bottlenecks (narrow essential reactions)
- Detect model errors (unexpectedly blocked reactions)

#### 3. Double Gene Knockout (Synthetic Lethality)

**Concept**: Two genes that are individually non-essential but lethal when deleted together.

**Algorithm**:
```python
wild_type_growth = FBA(model)

synthetic_lethal_pairs = []

for gene1, gene2 in all_gene_pairs:
    # Check individual knockouts
    ko1_growth = knockout_and_fba(gene1)
    ko2_growth = knockout_and_fba(gene2)

    # Both individually non-essential
    if ko1_growth > threshold and ko2_growth > threshold:

        # Test double knockout
        double_ko_growth = knockout_and_fba([gene1, gene2])

        # Synergy score
        expected = (ko1_growth * ko2_growth) / wild_type_growth
        observed = double_ko_growth
        synergy = expected - observed

        if double_ko_growth < threshold and synergy > 0.1:
            synthetic_lethal_pairs.append({
                'gene1': gene1,
                'gene2': gene2,
                'synergy_score': synergy
            })

return synthetic_lethal_pairs
```

**Synergy Score**:
```
synergy = (f₁ × f₂) - f₁₂

where:
  f₁ = growth fraction with gene1 KO
  f₂ = growth fraction with gene2 KO
  f₁₂ = growth fraction with both KO

High synergy → strong synthetic lethality
```

**Applications**:
- **Cancer therapy**: Target synthetic lethal pairs (one gene already defective in cancer)
- **Antibiotic combinations**: Design drug combinations
- **Metabolic engineering**: Find backup pathways

#### 4. Growth-Coupled Production

**Goal**: Engineer strains where product formation is coupled to cell growth.

**Algorithm**:
```python
target_metabolite = "ethanol"  # Production target

wild_type_growth = FBA(model)

candidates = []

for gene in model.genes:
    with model:
        gene.knock_out()

        # Maximize growth
        model.objective = "biomass"
        ko_growth = FBA(model)

        # Get production rate at optimal growth
        production_flux = model.reactions.get_by_id(
            f"EX_{target_metabolite}"
        ).flux

        # Check if production is coupled
        if production_flux > 0 and ko_growth > threshold:
            coupling_strength = production_flux / ko_growth

            candidates.append({
                'gene': gene.id,
                'growth': ko_growth,
                'production': production_flux,
                'coupling': coupling_strength
            })

return sorted(candidates, key=lambda x: x['coupling'], reverse=True)
```

**Coupling Metrics**:
- **Weak coupling**: Production possible without growth
- **Strong coupling**: Must produce to grow
- **Ideal target**: High production AND reasonable growth

**Verification**:
```python
# Test if production is truly required for growth
model.reactions.get_by_id(f"EX_{target}").lower_bound = 0  # Block export

growth_without_production = FBA(model)
# If growth == 0, production is essential (strongly coupled)
```

#### 5. Heterologous Pathway Design

**Workflow for non-native compound production**:

```
1. Target Selection
   └→ Compound not in native metabolism

2. Pathway Search
   └→ Find enzymatic route (KEGG, MetaCyc, literature)

3. Enzyme Selection
   └→ Choose genes from source organism

4. Model Modification
   ├→ Add new metabolites (formula, compartment)
   ├→ Add reactions (stoichiometry, bounds)
   ├→ Balance cofactors (NAD, ATP, etc.)
   └→ Add gene associations (heterologous genes)

5. Feasibility Testing
   └→ FBA: Can pathway produce target?

6. Optimization
   └→ Gene knockouts to enhance production

7. Strain Design
   └→ Integrate pathway + knockouts
```

**Example: 1,3-Propanediol from Glycerol**:

```python
from pathway_designer_tools import PathwayDesigner
import cobra

model = cobra.io.load_model("iML1515")
designer = PathwayDesigner(model)

# Step 1: Add new metabolites
hpa = designer.add_metabolite(
    "3hpald_c",
    "3-Hydroxypropionaldehyde",
    "C3H6O2",
    "c"
)
pdo = designer.add_metabolite(
    "13ppd_c",
    "1,3-Propanediol",
    "C3H8O2",
    "c"
)

# Step 2: Add heterologous reactions
# Reaction 1: Glycerol → 3-HPA (from K. pneumoniae)
designer.add_reaction(
    "DhaB",
    "Glycerol dehydratase",
    {
        model.metabolites.glyc_c: -1,
        hpa: 1,
        model.metabolites.h2o_c: 1
    },
    gene_reaction_rule="dhaB1 and dhaB2 and dhaB3"  # Heterologous
)

# Reaction 2: 3-HPA → 1,3-PDO (from K. pneumoniae)
designer.add_reaction(
    "DhaT",
    "1,3-propanediol oxidoreductase",
    {
        hpa: -1,
        model.metabolites.nadh_c: -1,
        model.metabolites.h_c: -1,
        pdo: 1,
        model.metabolites.nad_c: 1
    },
    gene_reaction_rule="dhaT"  # Heterologous
)

# Step 3: Add export reaction
designer.add_exchange_reaction(pdo)

# Step 4: Test feasibility
result = designer.test_pathway_feasibility("13ppd_c")
print(f"Production rate: {result['production_flux']:.3f} mmol/gDW/h")

# Step 5: Find knockouts to enhance production
# (Run single KO analysis with target = 13ppd_c)
```

### Workflow Diagram

```
┌─────────────────────────────────────────┐
│    Load Genome-Scale Metabolic Model    │
│  - SBML, JSON, or from BiGG database   │
│  - Validate model (mass balance, etc.)  │
└──────────────────┬──────────────────────┘
                   │
                   ▼
         ┌──────────────────┐
         │   Set Objective   │
         │  - Biomass (max)  │
         │  - Product (max)  │
         └─────────┬─────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
    ▼              ▼              ▼
┌────────┐  ┌────────────┐  ┌──────────┐
│Single  │  │  Essential │  │   FVA    │
│Gene KO │  │   Genes    │  │ Analysis │
└───┬────┘  └─────┬──────┘  └────┬─────┘
    │             │              │
    │             ▼              │
    │      ┌────────────┐        │
    │      │Double KO   │        │
    │      │ Synthetic  │        │
    │      │ Lethality  │        │
    │      └─────┬──────┘        │
    │            │               │
    └──────┬─────┴──────┬────────┘
           │            │
           ▼            ▼
    ┌─────────────────────────┐
    │  Growth-Coupled Prod    │
    │  - Target metabolite    │
    │  - Find KO for coupling │
    └──────────┬──────────────┘
               │
               ▼
    ┌──────────────────────┐
    │ Pathway Design (Opt) │
    │ - Add reactions      │
    │ - Add metabolites    │
    │ - Test feasibility   │
    └──────────┬───────────┘
               │
               ▼
       ┌───────────────┐
       │ Optimization  │
       │ - Rank targets│
       │ - Validate    │
       └───────┬───────┘
               │
               ▼
         ┌──────────┐
         │  OUTPUT  │
         │ CSV + PDF│
         └──────────┘
```

## Configuration Guide

The `config_example.yaml` file provides comprehensive configuration for all analysis parameters:

### Key Configuration Sections

#### 1. Model Configuration

```yaml
model:
  source: "bigg"          # or "file"
  bigg_id: "iML1515"      # E. coli model

  modifications:
    custom_bounds:
      EX_glc__D_e: [-10, 0]  # Glucose uptake
      EX_o2_e: [-20, 0]       # Oxygen uptake

    objective: "BIOMASS_Ec_iML1515_core_75p37M"
    medium: "default"         # or "minimal", "rich", "custom"
```

**Options**:
- `source: "bigg"` → Load from BiGG database (iML1515, Recon3D, etc.)
- `source: "file"` → Load from local SBML/JSON/MAT file
- `custom_bounds` → Set uptake rates, production limits
- `medium` → Predefined growth conditions

#### 2. Analysis Methods

```yaml
methods:
  enabled:
    - single
    - essential
    - fva
    - production  # Optional
    - pathway     # Optional

  single_knockout:
    growth_threshold: 0.05  # Essential if < 5% WT growth
    processes: -1           # Parallel processes

  fva:
    fraction_of_optimum: 0.9
    reactions: "all"  # or list specific reactions
```

**Method Selection**:
- Enable only needed methods (faster runtime)
- Adjust thresholds based on organism/application
- Use parallelization for large models

#### 3. Pathway Design Configuration

```yaml
methods:
  pathway_design:
    enabled: true
    target_compound: "13ppd_c"
    pathway_template: "1-3-propanediol"  # or "custom"

    custom_pathway:
      metabolites:
        - id: "new_met_c"
          name: "New Metabolite"
          formula: "C6H12O6"
          compartment: "c"

      reactions:
        - id: "NEW_RXN"
          name: "New Reaction"
          metabolites: {glc__D_c: -1, new_met_c: 1}
          gene_reaction_rule: "heterologous_gene"

    cofactor_balancing: true
    test_feasibility: true
```

**Pathway Templates**:
- `ethanol`: Ethanol production from pyruvate
- `succinate`: Enhanced succinate production
- `1-3-propanediol`: 1,3-PDO from glycerol
- `custom`: Define your own pathway

#### 4. Output and Visualization

```yaml
output:
  output_dir: "../metabolic_results"
  save_model: true  # Save modified model

visualization:
  enabled: true
  plots:
    growth_impact: true
    flux_distribution: true
    pathway_map: false  # Requires escher
```

### Example Configurations

**Fast testing (core model)**:
```yaml
model:
  source: "bigg"
  bigg_id: "textbook"  # Core model (fast)

methods:
  enabled:
    - single
    - fva
```

**Production strain design**:
```yaml
model:
  source: "bigg"
  bigg_id: "iML1515"
  modifications:
    custom_bounds:
      EX_glc__D_e: [-10, 0]
      EX_o2_e: [-20, 0]

methods:
  enabled:
    - single
    - fva
    - production

  production:
    enabled: true
    target_metabolite: "succ_c"
    min_production: 0.1
```

**Heterologous pathway engineering**:
```yaml
model:
  source: "bigg"
  bigg_id: "iML1515"

methods:
  enabled:
    - pathway
    - single
    - production

  pathway_design:
    enabled: true
    target_compound: "13ppd_c"
    pathway_template: "1-3-propanediol"
    test_feasibility: true

  production:
    enabled: true
    target_metabolite: "13ppd_c"
```

**Drug target discovery (human)**:
```yaml
model:
  source: "bigg"
  bigg_id: "Recon3D"

methods:
  enabled:
    - essential
    - double  # Synthetic lethality

  essential_genes:
    growth_threshold: 0.05

  double_knockout:
    growth_threshold: 0.05
    only_synthetic_lethal: true
```

## Citation

If you use this metabolic modeling module, please cite the Virtual Lab paper:

Swanson, K., Wu, W., Bulaong, N.L. et al. The Virtual Lab of AI agents designs new SARS-CoV-2 nanobodies. *Nature* (2025). https://doi.org/10.1038/s41586-025-09442-9

And the COBRApy paper:

Ebrahim, A., Lerman, J. A., Palsson, B. O., & Hyduke, D. R. (2013). COBRApy: COnstraints-Based Reconstruction and Analysis for Python. *BMC Systems Biology*, 7(1), 74.

## License

This module is part of the Virtual Lab project and is licensed under the MIT License.

## Contact

For questions or issues:
- Open an issue on GitHub
- Refer to the main Virtual Lab documentation
- Check COBRApy documentation
