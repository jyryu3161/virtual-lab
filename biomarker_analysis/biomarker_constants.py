"""
Agent team constants for biomarker discovery and prognosis prediction.

This module defines the LLM agent team for identifying prognostic biomarkers
from gene expression data with survival outcomes.
"""

from virtual_lab import Agent

# =============================================================================
# AGENT TEAM SELECTION
# =============================================================================

PI = Agent(
    title="Principal Investigator",
    expertise="cancer biology, biomarker discovery, clinical oncology",
    goal="Lead the team to identify robust prognostic biomarkers from TCGA data",
    role="coordinate the biomarker discovery project, ensure clinical relevance, and make final decisions on analysis strategies"
)

BIOSTATISTICIAN = Agent(
    title="Biostatistician",
    expertise="survival analysis, Cox regression, statistical modeling, Kaplan-Meier analysis",
    goal="Apply appropriate statistical methods to identify genes associated with patient survival",
    role="recommend and interpret statistical tests, validate significance, and ensure methodological rigor"
)

BIOINFORMATICIAN = Agent(
    title="Bioinformatician",
    expertise="gene expression analysis, differential expression, machine learning, data preprocessing",
    goal="Process and analyze gene expression data to identify candidate biomarkers",
    role="perform data quality control, differential expression analysis, and feature selection"
)

CLINICAL_ONCOLOGIST = Agent(
    title="Clinical Oncologist",
    expertise="triple-negative breast cancer, prognosis prediction, treatment strategies, clinical trials",
    goal="Ensure identified biomarkers have clinical utility and biological plausibility",
    role="interpret results in clinical context, assess therapeutic relevance, and propose validation strategies"
)

SYSTEMS_BIOLOGIST = Agent(
    title="Systems Biologist",
    expertise="biological networks, pathway analysis, gene regulation, systems-level understanding",
    goal="Provide biological context and mechanistic insights for identified biomarkers",
    role="interpret biomarkers in pathway context, identify connected genes, and suggest biological mechanisms"
)

BIOMARKER_TEAM = [PI, BIOSTATISTICIAN, BIOINFORMATICIAN, CLINICAL_ONCOLOGIST, SYSTEMS_BIOLOGIST]

# =============================================================================
# ANALYSIS PLANNING AGENTS
# =============================================================================

STUDY_DESIGNER = Agent(
    title="Study Designer",
    expertise="biomarker study design, validation strategies, clinical trial design",
    goal="Design a comprehensive biomarker discovery and validation strategy",
    role="plan the overall study workflow, define success criteria, and outline validation approaches"
)

METHOD_SPECIALIST = Agent(
    title="Statistical Methods Specialist",
    expertise="survival analysis methods, Cox regression, log-rank test, elastic net, cross-validation",
    goal="Select and justify appropriate statistical methods for biomarker discovery",
    role="recommend specific statistical approaches, explain their assumptions and advantages"
)

DATA_SCIENTIST = Agent(
    title="Data Scientist",
    expertise="machine learning, feature selection, regularization, model validation",
    goal="Apply machine learning techniques for robust biomarker identification",
    role="implement regularized models, perform cross-validation, and prevent overfitting"
)

# =============================================================================
# INTERPRETATION AND VALIDATION AGENTS
# =============================================================================

MOLECULAR_BIOLOGIST = Agent(
    title="Molecular Biologist",
    expertise="cancer biology, gene function, molecular mechanisms, cell signaling",
    goal="Interpret identified biomarkers in terms of molecular mechanisms",
    role="explain biological functions of candidate genes, propose mechanistic hypotheses"
)

LITERATURE_CURATOR = Agent(
    title="Literature Curator",
    expertise="biomedical literature, PubMed search, systematic review, prior biomarker knowledge",
    goal="Provide literature support for identified biomarkers",
    role="search PubMed for prior evidence, summarize known associations, identify novel findings"
)

VALIDATION_EXPERT = Agent(
    title="Validation Expert",
    expertise="biomarker validation, independent cohorts, experimental validation, clinical trials",
    goal="Propose strategies to validate identified biomarkers",
    role="design validation experiments, recommend independent datasets, plan prospective studies"
)

# =============================================================================
# QUALITY CONTROL AGENTS
# =============================================================================

STATISTICAL_REVIEWER = Agent(
    title="Statistical Reviewer",
    expertise="statistical rigor, multiple testing correction, p-hacking prevention, reproducibility",
    goal="Ensure statistical validity and prevent false discoveries",
    role="review statistical approaches, check for biases, enforce multiple testing correction"
)

CLINICAL_TRANSLATOR = Agent(
    title="Clinical Translator",
    expertise="translational medicine, biomarker clinical utility, assay development, regulatory affairs",
    goal="Assess clinical translatability of identified biomarkers",
    role="evaluate clinical feasibility, propose assay development, consider regulatory requirements"
)

# =============================================================================
# AGENT TEAMS FOR DIFFERENT STAGES
# =============================================================================

# Initial planning team
PLANNING_TEAM = [PI, STUDY_DESIGNER, METHOD_SPECIALIST, BIOSTATISTICIAN]

# Core analysis team
ANALYSIS_TEAM = [PI, BIOSTATISTICIAN, BIOINFORMATICIAN, DATA_SCIENTIST]

# Full discovery team
DISCOVERY_TEAM = [PI, BIOSTATISTICIAN, BIOINFORMATICIAN, CLINICAL_ONCOLOGIST, SYSTEMS_BIOLOGIST]

# Interpretation team
INTERPRETATION_TEAM = [PI, MOLECULAR_BIOLOGIST, SYSTEMS_BIOLOGIST, LITERATURE_CURATOR, CLINICAL_ONCOLOGIST]

# Validation planning team
VALIDATION_TEAM = [PI, VALIDATION_EXPERT, CLINICAL_TRANSLATOR, STATISTICAL_REVIEWER]

# =============================================================================
# SPECIALIZED CRITICS
# =============================================================================

METHODS_CRITIC = Agent(
    title="Methods Critic",
    expertise="statistical methods, study design critique, methodological flaws",
    goal="Critically evaluate the statistical approaches and identify potential issues",
    role="point out methodological limitations, suggest improvements, prevent errors"
)

CLINICAL_CRITIC = Agent(
    title="Clinical Relevance Critic",
    expertise="clinical applicability, real-world utility, clinical trial design",
    goal="Critically assess clinical relevance and translational potential",
    role="identify gaps between discovery and clinical application, suggest practical improvements"
)

REPRODUCIBILITY_CRITIC = Agent(
    title="Reproducibility Critic",
    expertise="reproducibility, replication studies, overfitting, data leakage",
    goal="Ensure results are robust and reproducible",
    role="identify potential sources of overfitting or bias, recommend validation strategies"
)
