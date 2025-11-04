"""
Pathway Design Tools for Metabolic Engineering

This module provides tools for designing and adding heterologous pathways
to genome-scale metabolic models.

Author: Virtual Lab
Date: 2025-11-04
"""

import cobra
from cobra import Model, Reaction, Metabolite
from typing import List, Dict, Tuple, Optional
import pandas as pd


class PathwayDesigner:
    """
    Tools for designing heterologous metabolic pathways.

    This class helps with:
    - Adding new reactions to models
    - Designing production pathways
    - Balancing cofactors
    - Checking pathway feasibility
    """

    def __init__(self, model: Model):
        self.model = model
        self.added_reactions = []

    def check_metabolite_exists(self, metabolite_id: str) -> bool:
        """Check if a metabolite exists in the model"""
        try:
            self.model.metabolites.get_by_id(metabolite_id)
            return True
        except KeyError:
            return False

    def check_reaction_exists(self, reaction_id: str) -> bool:
        """Check if a reaction exists in the model"""
        try:
            self.model.reactions.get_by_id(reaction_id)
            return True
        except KeyError:
            return False

    def add_metabolite(self,
                      metabolite_id: str,
                      name: str,
                      formula: str = "",
                      compartment: str = "c") -> Metabolite:
        """
        Add a new metabolite to the model.

        Args:
            metabolite_id: Metabolite ID
            name: Metabolite name
            formula: Chemical formula
            compartment: Compartment ('c' for cytoplasm, 'e' for extracellular)

        Returns:
            Metabolite object
        """
        if self.check_metabolite_exists(metabolite_id):
            print(f"  Metabolite {metabolite_id} already exists")
            return self.model.metabolites.get_by_id(metabolite_id)

        met = Metabolite(
            metabolite_id,
            formula=formula,
            name=name,
            compartment=compartment
        )
        self.model.add_metabolites([met])
        print(f"  ✓ Added metabolite: {metabolite_id} ({name})")
        return met

    def add_reaction(self,
                    reaction_id: str,
                    name: str,
                    metabolites: Dict[Metabolite, float],
                    lower_bound: float = 0,
                    upper_bound: float = 1000,
                    gene_reaction_rule: str = "") -> Reaction:
        """
        Add a new reaction to the model.

        Args:
            reaction_id: Reaction ID
            name: Reaction name
            metabolites: Dictionary of {metabolite: stoichiometry}
            lower_bound: Lower flux bound
            upper_bound: Upper flux bound
            gene_reaction_rule: Gene-protein-reaction rule (e.g., "gene1 and gene2")

        Returns:
            Reaction object
        """
        if self.check_reaction_exists(reaction_id):
            print(f"  Reaction {reaction_id} already exists")
            return self.model.reactions.get_by_id(reaction_id)

        rxn = Reaction(reaction_id)
        rxn.name = name
        rxn.lower_bound = lower_bound
        rxn.upper_bound = upper_bound
        rxn.add_metabolites(metabolites)

        if gene_reaction_rule:
            rxn.gene_reaction_rule = gene_reaction_rule

        self.model.add_reactions([rxn])
        self.added_reactions.append(reaction_id)

        print(f"  ✓ Added reaction: {reaction_id}")
        print(f"    {rxn.reaction}")
        print(f"    Bounds: [{lower_bound}, {upper_bound}]")
        if gene_reaction_rule:
            print(f"    Genes: {gene_reaction_rule}")

        return rxn

    def add_exchange_reaction(self,
                             metabolite: Metabolite,
                             lower_bound: float = 0,
                             upper_bound: float = 1000) -> Reaction:
        """
        Add an exchange reaction for a metabolite.

        Args:
            metabolite: Metabolite object
            lower_bound: Lower bound (0 = no uptake, negative = uptake allowed)
            upper_bound: Upper bound (secretion)

        Returns:
            Exchange reaction
        """
        exchange_id = f"EX_{metabolite.id}"

        if self.check_reaction_exists(exchange_id):
            print(f"  Exchange reaction {exchange_id} already exists")
            return self.model.reactions.get_by_id(exchange_id)

        return self.add_reaction(
            reaction_id=exchange_id,
            name=f"Exchange for {metabolite.name}",
            metabolites={metabolite: -1},
            lower_bound=lower_bound,
            upper_bound=upper_bound
        )

    def add_transport_reaction(self,
                               metabolite_in: Metabolite,
                               metabolite_out: Metabolite,
                               reaction_id: str = None) -> Reaction:
        """
        Add a transport reaction between compartments.

        Args:
            metabolite_in: Metabolite in source compartment
            metabolite_out: Metabolite in target compartment
            reaction_id: Custom reaction ID (auto-generated if None)

        Returns:
            Transport reaction
        """
        if reaction_id is None:
            reaction_id = f"TRANS_{metabolite_in.id}_{metabolite_out.id}"

        return self.add_reaction(
            reaction_id=reaction_id,
            name=f"Transport {metabolite_in.name}",
            metabolites={
                metabolite_in: -1,
                metabolite_out: 1
            },
            lower_bound=-1000,
            upper_bound=1000
        )

    def design_linear_pathway(self,
                            pathway_name: str,
                            substrates: List[str],
                            intermediates: List[Tuple[str, str, str]],  # (id, name, formula)
                            product: str,
                            gene_ids: List[str] = None) -> List[Reaction]:
        """
        Design a linear metabolic pathway.

        Args:
            pathway_name: Name of the pathway
            substrates: List of substrate metabolite IDs
            intermediates: List of (id, name, formula) for intermediate metabolites
            product: Product metabolite ID
            gene_ids: List of gene IDs for each step (optional)

        Returns:
            List of added reactions
        """
        print(f"\n=== Designing Linear Pathway: {pathway_name} ===")

        # Get substrate metabolites
        substrate_mets = []
        for sub_id in substrates:
            if self.check_metabolite_exists(sub_id):
                substrate_mets.append(self.model.metabolites.get_by_id(sub_id))
            else:
                print(f"  ERROR: Substrate {sub_id} not found in model")
                return []

        # Add intermediate metabolites
        intermediate_mets = []
        for met_id, met_name, formula in intermediates:
            met = self.add_metabolite(met_id, met_name, formula)
            intermediate_mets.append(met)

        # Check if product exists
        if self.check_metabolite_exists(product):
            product_met = self.model.metabolites.get_by_id(product)
        else:
            print(f"  ERROR: Product {product} not found in model")
            return []

        # Create reactions for each step
        reactions = []
        all_mets = substrate_mets + intermediate_mets + [product_met]

        for i in range(len(all_mets) - 1):
            rxn_id = f"{pathway_name}_step{i+1}"
            gene = gene_ids[i] if gene_ids and i < len(gene_ids) else ""

            rxn = self.add_reaction(
                reaction_id=rxn_id,
                name=f"{pathway_name} Step {i+1}",
                metabolites={
                    all_mets[i]: -1,
                    all_mets[i+1]: 1
                },
                lower_bound=0,
                upper_bound=1000,
                gene_reaction_rule=gene
            )
            reactions.append(rxn)

        # Add exchange for product
        self.add_exchange_reaction(product_met)

        print(f"\n✓ Pathway {pathway_name} added with {len(reactions)} reactions")
        return reactions

    def balance_cofactors(self,
                         reaction: Reaction,
                         cofactor_system: str = "NAD") -> Reaction:
        """
        Balance a reaction with appropriate cofactors.

        Args:
            reaction: Reaction to balance
            cofactor_system: Cofactor system (NAD, NADP, FAD, ATP)

        Returns:
            Modified reaction
        """
        # Common cofactors in E. coli models
        cofactor_map = {
            "NAD": ("nad_c", "nadh_c", "h_c"),      # NAD+ → NADH + H+
            "NADP": ("nadp_c", "nadph_c", "h_c"),   # NADP+ → NADPH + H+
            "FAD": ("fad_c", "fadh2_c"),            # FAD → FADH2
            "ATP": ("atp_c", "adp_c", "pi_c", "h_c")  # ATP → ADP + Pi + H+
        }

        if cofactor_system in cofactor_map:
            cofactors = cofactor_map[cofactor_system]

            # Add cofactors to reaction
            if cofactor_system == "ATP":
                reaction.add_metabolites({
                    self.model.metabolites.get_by_id(cofactors[0]): -1,  # ATP
                    self.model.metabolites.get_by_id(cofactors[1]): 1,   # ADP
                    self.model.metabolites.get_by_id(cofactors[2]): 1,   # Pi
                    self.model.metabolites.get_by_id(cofactors[3]): 1    # H+
                })
            else:
                reaction.add_metabolites({
                    self.model.metabolites.get_by_id(cofactors[0]): -1,
                    self.model.metabolites.get_by_id(cofactors[1]): 1
                })
                if len(cofactors) > 2:
                    reaction.add_metabolites({
                        self.model.metabolites.get_by_id(cofactors[2]): 1  # H+
                    })

            print(f"  ✓ Balanced with {cofactor_system}")
            print(f"    {reaction.reaction}")

        return reaction

    def test_pathway_feasibility(self, target_metabolite_id: str) -> Dict:
        """
        Test if a pathway can produce the target metabolite.

        Args:
            target_metabolite_id: ID of target metabolite

        Returns:
            Dictionary with feasibility results
        """
        print(f"\n=== Testing Pathway Feasibility ===")
        print(f"Target: {target_metabolite_id}")

        # Get exchange reaction
        exchange_id = f"EX_{target_metabolite_id}"

        if not self.check_reaction_exists(exchange_id):
            print(f"  ERROR: No exchange reaction for {target_metabolite_id}")
            return {"feasible": False, "error": "No exchange reaction"}

        exchange = self.model.reactions.get_by_id(exchange_id)

        # Set as objective
        old_objective = self.model.objective
        self.model.objective = exchange_id

        # Optimize
        solution = self.model.optimize()

        # Restore objective
        self.model.objective = old_objective

        results = {
            "feasible": solution.status == "optimal",
            "production_flux": solution.fluxes[exchange_id] if solution.status == "optimal" else 0,
            "growth_rate": solution.objective_value if solution.status == "optimal" else 0
        }

        if results["feasible"]:
            print(f"  ✓ Pathway is FEASIBLE")
            print(f"    Production flux: {results['production_flux']:.4f}")
            print(f"    With growth rate: {results['growth_rate']:.4f}")
        else:
            print(f"  ✗ Pathway is NOT feasible")
            print(f"    Status: {solution.status}")

        return results

    def get_summary(self) -> pd.DataFrame:
        """Get summary of added reactions"""
        if not self.added_reactions:
            return pd.DataFrame()

        summary = []
        for rxn_id in self.added_reactions:
            rxn = self.model.reactions.get_by_id(rxn_id)
            summary.append({
                "reaction_id": rxn_id,
                "name": rxn.name,
                "equation": rxn.reaction,
                "bounds": f"[{rxn.lower_bound}, {rxn.upper_bound}]",
                "genes": rxn.gene_reaction_rule
            })

        return pd.DataFrame(summary)


# Example pathway templates
EXAMPLE_PATHWAYS = {
    "ethanol": {
        "name": "Ethanol Production Pathway",
        "description": "Convert pyruvate to ethanol",
        "substrates": ["pyr_c"],
        "intermediates": [("acald_c", "Acetaldehyde", "C2H4O")],
        "product": "etoh_c",
        "steps": [
            "Pyruvate → Acetaldehyde (pyruvate decarboxylase)",
            "Acetaldehyde → Ethanol (alcohol dehydrogenase)"
        ]
    },
    "succinate": {
        "name": "Succinate Production Enhancement",
        "description": "Enhance succinate production from PEP",
        "substrates": ["pep_c", "co2_c"],
        "intermediates": [("oaa_c", "Oxaloacetate", "C4H4O5")],
        "product": "succ_c",
        "steps": [
            "PEP + CO2 → Oxaloacetate (PEP carboxylase)",
            "Oxaloacetate → Succinate (reductive TCA)"
        ]
    },
    "1-3-propanediol": {
        "name": "1,3-Propanediol Production",
        "description": "Heterologous pathway for 1,3-PDO from glycerol",
        "substrates": ["glyc_c"],
        "intermediates": [("3hpald_c", "3-Hydroxypropionaldehyde", "C3H6O2")],
        "product": "13ppd_c",
        "steps": [
            "Glycerol → 3-HPA (glycerol dehydratase)",
            "3-HPA → 1,3-PDO (1,3-propanediol oxidoreductase)"
        ]
    }
}
