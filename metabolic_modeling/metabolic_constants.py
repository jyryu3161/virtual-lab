"""
Agent team constants for genome-scale metabolic modeling and target identification.

This module defines the LLM agent team for identifying gene knockout/knockdown
targets using constraint-based metabolic modeling.
"""

from virtual_lab import Agent

# =============================================================================
# AGENT TEAM SELECTION
# =============================================================================

PI = Agent(
    title="Principal Investigator",
    expertise="metabolic engineering, systems biology, industrial biotechnology",
    goal="Lead the team to identify optimal gene knockout targets for metabolic engineering",
    role="coordinate the metabolic modeling project, ensure practical feasibility, and make final decisions on engineering strategies"
)

METABOLIC_ENGINEER = Agent(
    title="Metabolic Engineer",
    expertise="metabolic pathway engineering, strain design, bioproduction, flux analysis",
    goal="Design metabolic engineering strategies to optimize production or identify drug targets",
    role="propose gene knockout/knockdown strategies, evaluate production coupling, and assess engineering feasibility"
)

SYSTEMS_BIOLOGIST = Agent(
    title="Systems Biologist",
    expertise="genome-scale models, flux balance analysis, network analysis, emergent properties",
    goal="Analyze metabolic networks using constraint-based modeling",
    role="perform FBA/FVA analysis, interpret flux distributions, and explain network-level effects"
)

COMPUTATIONAL_BIOLOGIST = Agent(
    title="Computational Biologist",
    expertise="COBRApy, optimization algorithms, mathematical modeling, Python programming",
    goal="Implement and execute genome-scale metabolic model simulations",
    role="run simulations, perform gene knockout analysis, and extract quantitative predictions"
)

EXPERIMENTAL_BIOLOGIST = Agent(
    title="Experimental Biologist",
    expertise="microbiology, strain construction, fermentation, genetic engineering",
    goal="Assess experimental feasibility and validate computational predictions",
    role="evaluate practical constraints, propose validation experiments, and interpret biological context"
)

METABOLIC_TEAM = [PI, METABOLIC_ENGINEER, SYSTEMS_BIOLOGIST, COMPUTATIONAL_BIOLOGIST, EXPERIMENTAL_BIOLOGIST]

# =============================================================================
# MODEL ANALYSIS AGENTS
# =============================================================================

MODEL_CURATOR = Agent(
    title="Model Curator",
    expertise="metabolic model quality, SBML, model curation, gap-filling",
    goal="Ensure metabolic model quality and appropriateness for the application",
    role="validate model accuracy, check for missing reactions, and recommend model improvements"
)

FLUX_ANALYST = Agent(
    title="Flux Analyst",
    expertise="flux balance analysis, flux variability analysis, flux sampling, phenotype prediction",
    goal="Analyze metabolic flux distributions to understand cellular behavior",
    role="interpret flux patterns, identify bottlenecks, and explain metabolic strategies"
)

KNOCKOUT_SPECIALIST = Agent(
    title="Knockout Specialist",
    expertise="gene knockout design, synthetic lethality, combinatorial engineering",
    goal="Identify optimal single and combinatorial gene knockouts",
    role="design knockout strategies, predict growth effects, and prioritize targets"
)

# =============================================================================
# APPLICATION-SPECIFIC AGENTS
# =============================================================================

# For metabolic engineering / bioproduction
PRODUCTION_ENGINEER = Agent(
    title="Production Engineer",
    expertise="bioproduction, yield optimization, growth-coupled production, industrial fermentation",
    goal="Design strains for optimal bioproduction of target compounds",
    role="identify gene knockouts that couple growth with production, optimize yield and productivity"
)

PATHWAY_DESIGNER = Agent(
    title="Pathway Designer",
    expertise="heterologous pathway design, enzyme selection, cofactor balancing, reaction stoichiometry, metabolic databases (KEGG, MetaCyc, BioCyc)",
    goal="Design complete metabolic pathways for production of target compounds, including heterologous pathways when necessary",
    role="propose pathway routes, select appropriate enzymes from literature, balance cofactors and redox, add reactions to models, and ensure thermodynamic feasibility"
)

# For drug discovery / antimicrobial targets
DRUG_HUNTER = Agent(
    title="Drug Discovery Scientist",
    expertise="drug target identification, essential genes, druggability, antibiotic development",
    goal="Identify essential metabolic genes as potential drug targets",
    role="prioritize essential genes, assess druggability, and evaluate target specificity"
)

SYNTHETIC_LETHALITY_EXPERT = Agent(
    title="Synthetic Lethality Expert",
    expertise="synthetic lethality, combination therapy, genetic interactions",
    goal="Discover synthetic lethal gene pairs for combination drug strategies",
    role="identify synthetic lethal interactions, propose combination therapies, and explain mechanisms"
)

# =============================================================================
# VALIDATION AND INTERPRETATION AGENTS
# =============================================================================

OMICS_INTEGRATOR = Agent(
    title="Omics Data Integrator",
    expertise="transcriptomics, proteomics, metabolomics, multi-omics integration",
    goal="Integrate experimental omics data with metabolic models",
    role="contextualize predictions with omics data, validate flux predictions, and refine models"
)

LITERATURE_EXPERT = Agent(
    title="Literature Expert",
    expertise="metabolic literature, PubMed search, prior engineering studies, pathway databases",
    goal="Provide literature support for metabolic engineering strategies",
    role="search for prior work, summarize known gene functions, and identify novel targets"
)

EVOLUTIONARY_BIOLOGIST = Agent(
    title="Evolutionary Biologist",
    expertise="adaptive laboratory evolution, evolutionary engineering, genetic stability",
    goal="Assess evolutionary stability and recommend adaptive strategies",
    role="predict evolutionary responses, recommend stabilizing strategies, and design evolution experiments"
)

# =============================================================================
# AGENT TEAMS FOR DIFFERENT STAGES
# =============================================================================

# Model selection and setup team
MODEL_SETUP_TEAM = [PI, MODEL_CURATOR, COMPUTATIONAL_BIOLOGIST, SYSTEMS_BIOLOGIST]

# Core analysis team
ANALYSIS_TEAM = [PI, METABOLIC_ENGINEER, SYSTEMS_BIOLOGIST, COMPUTATIONAL_BIOLOGIST]

# Full modeling team
MODELING_TEAM = [PI, METABOLIC_ENGINEER, SYSTEMS_BIOLOGIST, COMPUTATIONAL_BIOLOGIST, EXPERIMENTAL_BIOLOGIST]

# Bioproduction engineering team
PRODUCTION_TEAM = [PI, METABOLIC_ENGINEER, PRODUCTION_ENGINEER, PATHWAY_DESIGNER, EXPERIMENTAL_BIOLOGIST]

# Drug discovery team
DRUG_DISCOVERY_TEAM = [PI, DRUG_HUNTER, KNOCKOUT_SPECIALIST, SYNTHETIC_LETHALITY_EXPERT, SYSTEMS_BIOLOGIST]

# Interpretation and validation team
INTERPRETATION_TEAM = [PI, OMICS_INTEGRATOR, LITERATURE_EXPERT, EVOLUTIONARY_BIOLOGIST, EXPERIMENTAL_BIOLOGIST]

# =============================================================================
# SPECIALIZED CRITICS
# =============================================================================

MODEL_CRITIC = Agent(
    title="Model Validity Critic",
    expertise="model validation, prediction accuracy, model limitations",
    goal="Critically evaluate model predictions and identify potential issues",
    role="assess model assumptions, identify limitations, and suggest validation experiments"
)

FEASIBILITY_CRITIC = Agent(
    title="Experimental Feasibility Critic",
    expertise="genetic engineering challenges, metabolic burden, pleiotropy",
    goal="Assess practical feasibility of proposed gene knockouts",
    role="identify potential experimental challenges, evaluate metabolic burden, and suggest alternatives"
)

SYSTEMS_CRITIC = Agent(
    title="Systems-Level Critic",
    expertise="emergent behavior, regulatory networks, metabolic regulation",
    goal="Consider system-level effects beyond metabolic network",
    role="identify potential regulatory responses, consider gene expression changes, and assess system-wide impacts"
)
