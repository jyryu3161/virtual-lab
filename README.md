# Virtual Lab

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/virtual-lab)](https://badge.fury.io/py/virtual-lab)
[![PyPI version](https://badge.fury.io/py/virtual-lab.svg)](https://badge.fury.io/py/virtual-lab)
[![Downloads](https://pepy.tech/badge/virtual-lab)](https://pepy.tech/project/virtual-lab)
[![license](https://img.shields.io/github/license/zou-group/virtual-lab.svg)](https://github.com/zou-group/virtual-lab/blob/main/LICENSE.txt)

![Virtual Lab](images/virtual_lab_architecture.png)

The **Virtual Lab** is an AI-human collaboration for science research. In the Virtual Lab, a human researcher works with a team of large language model (LLM) **agents** to perform scientific research. Interaction between the human researcher and the LLM agents occurs via a series of **team meetings**, where all the LLM agents discuss a scientific agenda posed by the human researcher, and **individual meetings**, where the human researcher interacts with a single LLM agent to solve a particular scientific task.

Please see our paper [The Virtual Lab of AI agents designs new SARS-CoV-2 nanobodies](https://www.nature.com/articles/s41586-025-09442-9) for more details on the Virtual Lab and an application to nanobody design for SARS-CoV-2.

If you use the Virtual Lab, please cite our work as follows:

Swanson, K., Wu, W., Bulaong, N.L. et al. The Virtual Lab of AI agents designs new SARS-CoV-2 nanobodies. *Nature* (2025). https://doi.org/10.1038/s41586-025-09442-9


## Applications

### Virtual Lab for nanobody design

As a real-world demonstration, we applied the Virtual Lab to design nanobodies for one of the latest variants of SARS-CoV-2 (see [nanobody_design](https://github.com/zou-group/virtual-lab/tree/main/nanobody_design)). The Virtual Lab built a computational pipeline consisting of [ESM](https://www.science.org/doi/10.1126/science.ade2574), [AlphaFold-Multimer](https://www.biorxiv.org/content/10.1101/2021.10.04.463034v2), and [Rosetta](https://rosettacommons.org/software/) and used it to design 92 nanobodies that were experimentally validated.

Please see the notebook [nanobody_design/run_nanobody_design.ipynb](https://github.com/zou-group/virtual-lab/blob/main/nanobody_design/run_nanobody_design.ipynb) for an example of how to use the Virtual Lab to create agents and run team and individual meetings.

### Biomarker Analysis for Prognosis Prediction

The Virtual Lab enables **AI-guided biomarker discovery** where LLM agents collaborate to identify prognostic genes from gene expression data. The agent team discusses strategies, selects methods, interprets results, and plans validation—mimicking a real research team.

#### Features

- **AI Agent Team**: Biostatistician, Bioinformatician, Clinical Oncologist, Systems Biologist
- **Collaborative Analysis**: Agents discuss and justify statistical approaches
- **Multiple Statistical Methods**: Cox Proportional Hazards, Log-rank test, Differential Expression, Elastic Net
- **Literature Integration**: Agents search PubMed for biological context
- **Clinical Interpretation**: AI provides clinically-relevant insights
- **Comprehensive Visualization**: Volcano plots, Kaplan-Meier curves, heatmaps

#### Quick Start

**Using YAML Configuration (Recommended for Reproducibility):** 🆕

```bash
cd biomarker_analysis
cp config_example.yaml my_analysis.yaml
# Edit my_analysis.yaml to set your parameters
python scripts/run_with_config.py my_analysis.yaml
```

All parameters (data files, methods, thresholds) in one place!

**AI-Guided Discovery (Recommended for Scientific Insights):**

```bash
# Launch agent-based biomarker discovery workflow
jupyter notebook biomarker_analysis/run_biomarker_discovery.ipynb
```

The agents will:
1. Plan the analysis strategy (statistical methods, validation)
2. Execute biomarker discovery with selected methods
3. Interpret results with biological/clinical context
4. Search PubMed for literature support
5. Propose validation experiments

**Direct Analysis (Command Line):**

```bash
# Run the complete biomarker analysis directly
pixi run biomarker-analysis

# Or use Python directly
cd biomarker_analysis/scripts

python select_marker_genes.py \
    --input_file ../../Example_TCGA_TNBC_data.csv \
    --output_dir ../../biomarker_results \
    --n_top_genes 50 \
    --methods cox logrank differential elasticnet \
    --visualization
```

**Learning Tutorial:**

```bash
# Interactive tutorial (no agents, educational)
jupyter notebook biomarker_analysis/tutorial_biomarker_selection.ipynb
```

#### Example Dataset

The repository includes an example TCGA triple-negative breast cancer (TNBC) dataset:

- **File**: `Example_TCGA_TNBC_data.csv`
- **Samples**: 144 patients
- **Features**: ~20,000 gene expression values (log-transformed)
- **Outcomes**: Overall survival (OS) and survival time (OS.year)

#### Output Files

The analysis generates:

- `consensus_genes.csv`: Genes ranked by appearance across methods
- `{method}_results.csv`: Detailed results for each statistical method
- `top_biomarkers_summary.csv`: Comprehensive summary of top genes
- `marker_gene_analysis.pdf`: Visualization plots

#### Statistical Methods

1. **Cox Proportional Hazards Regression**
   - Models the relationship between gene expression and survival time
   - Accounts for censored data
   - Outputs hazard ratios (HR) and confidence intervals

2. **Log-rank Test (Kaplan-Meier Analysis)**
   - Compares survival curves between high vs. low expression groups
   - Non-parametric test for survival differences
   - Generates Kaplan-Meier plots

3. **Differential Expression Analysis**
   - Compares expression between event and censored groups
   - Uses Mann-Whitney U test (non-parametric)
   - Calculates fold changes and significance

4. **Elastic Net Cox Regression**
   - Regularized Cox model with L1/L2 penalties
   - Performs automatic feature selection
   - Identifies sparse set of prognostic genes

#### Tutorial

For a comprehensive guide on biomarker selection, see the tutorial notebook:
[biomarker_analysis/tutorial_biomarker_selection.ipynb](biomarker_analysis/tutorial_biomarker_selection.ipynb)

The tutorial covers:
- Data loading and exploration
- Each statistical method in detail
- Consensus gene identification
- Result visualization and interpretation
- Exporting results for further analysis

### Genome-Scale Metabolic Modeling for Metabolic Engineering

The Virtual Lab enables **AI-guided metabolic engineering** where LLM agents collaborate to design production strains and optimize metabolic pathways. The agent team designs knockout strategies, interprets flux distributions, engineers heterologous pathways, and plans validation—bringing systems biology expertise to industrial strain development.

#### Features

- **AI Agent Team**: Metabolic Engineer, Pathway Designer, Production Engineer, Systems Biologist
- **Production Strain Design**: Agents optimize gene knockouts to enhance product yield
- **Heterologous Pathway Engineering**: Design and add non-native production pathways
- **Constraint-Based Modeling**: Flux Balance Analysis (FBA) and Flux Variability Analysis (FVA) with COBRApy
- **Growth-Coupled Production**: Engineer strains where product formation is coupled to growth
- **Literature Integration**: Agents search PubMed for metabolic engineering strategies
- **Experimental Planning**: AI proposes strain construction and validation protocols
- **Multiple Organisms**: E. coli (biofuels/chemicals), Yeast (complex molecules), CHO cells (therapeutics)

#### Quick Start

**Using YAML Configuration (Recommended for Reproducibility):** 🆕

```bash
cd metabolic_modeling
cp config_example.yaml my_analysis.yaml
# Edit my_analysis.yaml to set model, methods, pathway design
python scripts/run_with_config.py my_analysis.yaml
```

All parameters (model selection, methods, pathway design) in one place!

**AI-Guided Metabolic Engineering (Recommended for Scientific Insights):**

```bash
# Launch agent-based metabolic engineering workflow
jupyter notebook metabolic_modeling/run_metabolic_engineering.ipynb

# Or with heterologous pathway design
jupyter notebook metabolic_modeling/run_metabolic_engineering_with_pathway.ipynb
```

The agents will:
1. Plan the metabolic engineering project (goals, model selection)
2. Execute genome-scale metabolic simulations (FBA, gene knockouts)
3. Interpret flux distributions and knockout effects
4. Design specific engineered strains
5. Search PubMed for gene functions and prior work
6. Plan experimental validation protocols

**Direct Analysis (Command Line):**

```bash
# Run metabolic analysis with E. coli core model (fast)
pixi run metabolic-analysis

# Run with larger E. coli iML1515 model
pixi run metabolic-iml1515

# Or use Python directly
cd metabolic_modeling/scripts

python metabolic_target_finder.py \
    --model_id textbook \
    --output_dir ../../metabolic_results \
    --ko_methods single essential fva \
    --visualization
```

**Learning Tutorial:**

```bash
# Interactive COBRApy tutorial (no agents, educational)
jupyter notebook metabolic_modeling/tutorial_metabolic_modeling.ipynb
```

#### Available Organisms for Engineering

**E. coli (Industrial workhorse):**
- `textbook`: Core model (95 reactions, 72 genes) - Fast, for learning
- `iML1515`: Latest model (2,712 reactions, 1,877 genes) - Production strain design
- Applications: Biofuels, platform chemicals, proteins, enzymes

**Yeast (Eukaryotic platform):**
- `iMM904`: S. cerevisiae (1,577 reactions, 904 genes) - Yeast engineering
- Applications: Ethanol, complex molecules, therapeutic proteins

**Mammalian Cells (Biopharmaceuticals):**
- `Recon3D`: Human metabolism (13,543 reactions, 3,288 genes) - CHO cell analogue
- Applications: Therapeutic proteins, antibodies, vaccines

**Custom Models:**
- Load from SBML, JSON, or MAT files
- Place in `metabolic_modeling/models/` directory

#### Engineering Methods

1. **Gene Knockout Optimization**
   - Identify deletions that enhance production
   - Test single and synergistic double knockouts
   - Ensure strain viability (avoid essential genes)

2. **Growth-Coupled Production**
   - Engineer obligate coupling between growth and product formation
   - Maximize yield and productivity
   - Enable evolutionary stability

3. **Heterologous Pathway Design**
   - Add non-native production pathways
   - Select enzymes from literature (PubMed search)
   - Balance cofactors and metabolic load

4. **Flux Bottleneck Identification**
   - Find rate-limiting steps using FVA
   - Target amplification candidates
   - Optimize metabolic flux distribution

5. **Strain Validation Planning**
   - AI agents propose experimental protocols
   - Predict phenotypes for validation
   - Design growth and production assays

#### Output Files

The analysis generates:

- `model_summary.csv`: Model statistics and information
- `single_knockout_results.csv`: Growth effects of all gene knockouts
- `essential_genes_results.csv`: Essential genes with detailed annotations
- `double_knockout_results.csv`: Synthetic lethal gene pairs
- `fva_results.csv`: Flux ranges for all reactions
- `growth_coupled_results.csv`: Production-enhancing knockouts
- `metabolic_analysis.pdf`: Comprehensive visualization plots

#### Metabolic Engineering Use Cases

**Biofuel Production (Ethanol):**
```bash
# Optimize E. coli for ethanol from glucose
python metabolic_target_finder.py \
    --model_id iML1515 \
    --ko_methods single production fva \
    --target_metabolite etoh_c \
    --output_dir ethanol_production
```

**Platform Chemical Production (Succinate):**
```bash
# Design succinate production strain
python metabolic_target_finder.py \
    --model_id iML1515 \
    --ko_methods single production \
    --target_metabolite succ_c \
    --output_dir succinate_production
```

**Heterologous Pathway Engineering (1,3-Propanediol):**
```bash
# Add non-native pathway and optimize
cd metabolic_modeling
python scripts/run_with_config.py configs/pdo_production.yaml
# Includes pathway design from K. pneumoniae + knockout optimization
```

#### Tutorial

For a comprehensive guide on metabolic modeling, see the tutorial notebook:
[metabolic_modeling/tutorial_metabolic_modeling.ipynb](metabolic_modeling/tutorial_metabolic_modeling.ipynb)

The tutorial covers:
- Loading and analyzing metabolic models
- Flux Balance Analysis (FBA)
- Single and double gene knockout simulations
- Essential gene identification
- Flux Variability Analysis (FVA)
- Growth-coupled production strategies
- Working with large genome-scale models

#### Heterologous Pathway Design

The Virtual Lab includes **AI-guided pathway design** capabilities for engineering strains to produce compounds not native to the organism. The PATHWAY_DESIGNER agent collaborates with the team to design complete heterologous pathways, select enzymes from literature, and add reactions to metabolic models.

**Features:**
- Design multi-step heterologous pathways
- Add new metabolites and reactions to models programmatically
- Automatic cofactor balancing (NAD, NADP, FAD, ATP)
- Literature-guided enzyme selection from source organisms
- Pathway feasibility testing with FBA
- Integration with gene knockout optimization

**Complete Workflow with Pathway Design:**

```bash
# Launch AI-guided metabolic engineering with pathway design
jupyter notebook metabolic_modeling/run_metabolic_engineering_with_pathway.ipynb
```

The workflow demonstrates:
1. **Target Compound Selection**: Agents discuss what to produce (e.g., 1,3-propanediol)
2. **Pathway Design**: PATHWAY_DESIGNER identifies required enzymes and reactions from literature
3. **Model Modification**: Add new metabolites, reactions, and gene associations
4. **Feasibility Testing**: Verify the pathway can produce the target
5. **Knockout Optimization**: Identify gene deletions to enhance production
6. **Strain Design**: Combine heterologous genes + knockouts for final strain

**Example: 1,3-Propanediol Production**

```python
from pathway_designer_tools import PathwayDesigner
import cobra

# Load E. coli model
model = cobra.io.load_model("textbook")

# Initialize pathway designer
designer = PathwayDesigner(model)

# Add 1,3-PDO pathway from Klebsiella pneumoniae
# Step 1: Glycerol → 3-Hydroxypropionaldehyde (glycerol dehydratase)
# Step 2: 3-HPA → 1,3-Propanediol (1,3-propanediol oxidoreductase)

# Add metabolites
hpa = designer.add_metabolite("3hpald_c", "3-Hydroxypropionaldehyde", "C3H6O2", "c")
pdo = designer.add_metabolite("13ppd_c", "1,3-Propanediol", "C3H8O2", "c")

# Add reactions with gene associations
designer.add_reaction(
    reaction_id="PDO_DhaB",
    name="Glycerol dehydratase",
    metabolites={glyc: -1, hpa: 1, h2o: 1},
    gene_reaction_rule="dhaB1 and dhaB2 and dhaB3"
)

designer.add_reaction(
    reaction_id="PDO_DhaT",
    name="1,3-propanediol oxidoreductase",
    metabolites={hpa: -1, nadh: -1, h: -1, pdo: 1, nad: 1},
    gene_reaction_rule="dhaT"
)

# Test pathway feasibility
feasibility = designer.test_pathway_feasibility("13ppd_c")
print(f"Production rate: {feasibility['production_flux']:.4f} mmol/gDW/h")
```

**Available Example Pathways:**
- `ethanol`: Ethanol production from pyruvate
- `succinate`: Enhanced succinate production
- `1-3-propanediol`: 1,3-PDO from glycerol (heterologous)

See [metabolic_modeling/scripts/pathway_designer_tools.py](metabolic_modeling/scripts/pathway_designer_tools.py) for the complete API.


## Installation

The Virtual Lab can be installed using pip, conda, or pixi. Installation should only take a couple of minutes.

### Option 1: Quick Install with Pixi (Recommended)

[Pixi](https://pixi.sh) is a fast, modern package manager that handles all dependencies automatically. This is the **recommended installation method** for the best out-of-the-box experience.

#### Install Pixi

```bash
# On Linux and macOS
curl -fsSL https://pixi.sh/install.sh | bash

# On Windows
iwr -useb https://pixi.sh/install.ps1 | iex
```

#### Install Virtual Lab

```bash
git clone https://github.com/zou-group/virtual_lab.git
cd virtual_lab

# Install all dependencies (one command!)
pixi install

# Activate the environment
pixi shell
```

That's it! All dependencies including Python, Jupyter, and all scientific packages are now installed and ready to use.

#### Available Pixi Commands

```bash
# Run biomarker analysis
pixi run biomarker-analysis

# Launch tutorial notebook
pixi run biomarker-tutorial

# Launch Jupyter Lab
pixi run lab

# Run nanobody design notebook
pixi run nanobody-design

# Quick test of biomarker analysis
pixi run test-biomarker

# Clean up generated files
pixi run clean
```

#### Pixi Environments

The Virtual Lab provides several pre-configured environments:

- `default`: Core Virtual Lab functionality
- `dev`: Development tools (pytest, black, ruff, mypy)
- `nanobody`: Nanobody design dependencies
- `biomarker`: Biomarker analysis dependencies
- `full`: All features combined

To use a specific environment:

```bash
# Use biomarker environment
pixi shell -e biomarker

# Use full environment with all features
pixi shell -e full
```

### Option 2: Install with Conda

Create a conda environment and install dependencies:

```bash
conda create -y -n virtual_lab python=3.12
conda activate virtual_lab

# Install Virtual Lab
pip install virtual-lab

# For biomarker analysis, install additional dependencies
pip install lifelines scikit-survival pandas scipy matplotlib seaborn
```

### Option 3: Install with Pip (Minimal)

```bash
# Install from PyPI
pip install virtual-lab

# Or install from source
git clone https://github.com/zou-group/virtual_lab.git
cd virtual_lab
pip install -e .
```

### Install Optional Dependencies

For specific use cases, you may need additional packages:

```bash
# For nanobody design
pip install -e ".[nanobody-design]"

# For biomarker analysis
pip install lifelines scikit-survival statsmodels
```

### Verify Installation

```bash
# Test Python import
python -c "import virtual_lab; print('Virtual Lab successfully installed!')"

# Check Jupyter is available
jupyter --version

# For biomarker analysis, test dependencies
python -c "import lifelines, sksurv; print('Biomarker analysis dependencies installed!')"
```

## OpenAI API Key

The Virtual Lab currently uses GPT-4o from OpenAI. Save your OpenAI API key as the environment variable `OPENAI_API_KEY`.

```bash
# Add to your shell profile (.bashrc, .bash_profile, or .zshrc)
export OPENAI_API_KEY='your-api-key-here'

# Or set for current session
export OPENAI_API_KEY='your-api-key-here'
```

To get an OpenAI API key:
1. Visit https://platform.openai.com/api-keys
2. Sign up or log in
3. Create a new API key
4. Copy and save it securely
