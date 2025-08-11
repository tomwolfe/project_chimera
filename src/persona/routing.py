"""
Dynamic persona routing system that selects appropriate personas
based on prompt analysis and intermediate results.
"""

from typing import List, Dict, Set, Optional, Any
import re # Import re for regular expressions
import json
from pathlib import Path
import logging
from functools import lru_cache # Import lru_cache for caching

from src.models import PersonaConfig
from src.constants import SELF_ANALYSIS_KEYWORDS, SELF_ANALYSIS_PERSONA_SEQUENCE
from src.constants import is_self_analysis_prompt # Import the function for prompt analysis

logger = logging.getLogger(__name__)

class PersonaRouter:
    """Determines the optimal sequence of personas for a given prompt."""
    
    # Modified __init__ to accept persona_sets
    def __init__(self, all_personas: Dict[str, PersonaConfig], persona_sets: Dict[str, List[str]]):
        self.all_personas = all_personas
        self.persona_sets = persona_sets # Store persona sets
        
        self.domain_keywords = {
            "architecture": {
                "positive": [
                    "software architect", "system design", "code structure", "architecture pattern", 
                    "scalab", "perform", "modular", "refactor", "system", "structure", "database", 
                    "api", "framework", "codebase", "maintainability", "technical debt", 
                    "separation of concerns", "microservice", "monolith", "backend", "frontend"
                ],
                "negative": [
                    "building", "house", "construct", "physical", "brick", "concrete", 
                    "skyscraper", "residential", "commercial", "architecture firm", "civil engineer",
                    "urban planning", "interior design"
                ]
            },
            "security": {
                "positive": [
                    "vulnerab", "security", "exploit", "hack", "auth", "encrypt", "threat", 
                    "risk", "malware", "penetration", "compliance", "firewall", "ssl", "tls"
                ],
                "negative": []
            },
            "testing": {
                "positive": [
                    "test", "cover", "unit", "integration", "bug", "error", "quality", 
                    "qa", "defect", "debug", "validate", "assertion", "failure", "edge case"
                ],
                "negative": []
            },
            "devops": {
                "positive": [
                    "deploy", "ci/cd", "pipeline", "infra", "monitor", "cloud", "docker", 
                    "k8s", "ops", "server", "automation", "release", "scalability", 
                    "reliability", "performance", "logging", "alerting"
                ],
                "negative": []
            },
            "scientific": {
                "positive": ["scientific", "experiment", "hypothesis", "research", "data"],
                "negative": []
            },
            "business": {
                "positive": ["market", "business", "financial", "economy", "strategy"],
                "negative": []
            },
            "creative": {
                "positive": ["creative", "write", "story", "poem", "artistic"],
                "negative": []
            }
        }
        
        self.trigger_keywords = {
            "Security_Auditor": ["vulnerab", "security", "exploit", "hack", "auth", "encrypt", "threat", "risk", "malware", "penetration", "compliance", "attack vector", "data breach"],
            "Test_Engineer": ["test", "bug", "error", "quality", "coverage", "unit", "integration", "qa", "defect", "debug", "validate", "assertion", "failure", "edge case"],
            "DevOps_Engineer": ["deploy", "ci/cd", "pipeline", "infra", "monitor", "cloud", "docker", "k8s", "ops", "server", "automation", "release", "scalability", "reliability", "performance", "logging", "alerting"],
            "Code_Architect": ["architect", "design", "pattern", "scalab", "perform", "modular", "refactor", "system", "structure", "database", "api", "framework", "codebase", "maintainability", "technical debt", "separation of concerns"],
            "Constructive_Critic": ["improve", "refine", "optimize", "recommend", "suggest", "enhanc", "fix", "best practice"],
            "Skeptical_Generator": ["risk", "flaw", "limitation", "vulnerab", "bottleneck", "edge case", "failure point", "concern", "doubt"]
        }
    
    # The determine_domain method is not directly used for the main sequence, but can remain.
    # def determine_domain(self, prompt: str) -> str:
    #     # ... (existing logic)

    # The _analyze_prompt_domain method can remain if used elsewhere, but not for base sequence.
    # def _analyze_prompt_domain(self, prompt: str) -> Set[str]:
    #     # ... (existing logic)

    # REMOVED: _get_domain_specific_personas as it's replaced by direct use of persona_sets
    # def _get_domain_specific_personas(self, domains: Set[str]) -> List[str]:
    #     # ... (existing logic)

    def is_self_analysis_prompt(self, prompt: str) -> bool:
       """Standardized method to detect self-analysis prompts using central constants"""
       return is_self_analysis_prompt(prompt)

    def _apply_dynamic_adjustment(self, sequence: List[str], intermediate_results: Optional[Dict[str, Any]], prompt_lower: str) -> List[str]:
        """Apply dynamic adjustments to persona sequence based on intermediate results quality metrics."""
        if not intermediate_results:
            intermediate_results = {}
        
        quality_metrics = {}
        for step_name, result in intermediate_results.items():
            if isinstance(result, dict):
                if 'quality_metrics' in result and isinstance(result['quality_metrics'], dict):
                    for metric_name, value in result['quality_metrics'].items():
                        quality_metrics[metric_name] = max(quality_metrics.get(metric_name, 0.0), value)
                elif step_name.endswith("_Output") and isinstance(result, dict):
                    if "quality_metrics" in result and isinstance(result["quality_metrics"], dict):
                        for metric_name, value in result["quality_metrics"].items():
                            quality_metrics[metric_name] = max(quality_metrics.get(metric_name, 0.0), value)

        adjusted_sequence = sequence.copy()
        
        # --- Dynamic Adjustments based on Quality Metrics ---
        # These are heuristics and can be tuned.
        if quality_metrics.get('code_quality', 1.0) < 0.7:
            self._insert_persona_before_arbitrator(adjusted_sequence, 'Code_Architect')
            self._insert_persona_before_arbitrator(adjusted_sequence, 'Security_Auditor')
            self._insert_persona_before_arbitrator(adjusted_sequence, 'Test_Engineer')
        
        if quality_metrics.get('reasoning_depth', 1.0) < 0.6:
            self._insert_persona_before_arbitrator(adjusted_sequence, 'Devils_Advocate')
        
        if quality_metrics.get('test_coverage_estimate', 1.0) < 0.5:
            self._insert_persona_before_arbitrator(adjusted_sequence, 'Test_Engineer')
        
        if quality_metrics.get('security_risk_score', 0.0) > 0.7:
            self._insert_persona_before_arbitrator(adjusted_sequence, 'Security_Auditor')
            self._insert_persona_before_arbitrator(adjusted_sequence, 'DevOps_Engineer')

        # --- Dynamic Adjustments based on Prompt Misclassification ---
        if "Code_Architect" in adjusted_sequence:
            if ("building architect" in prompt_lower or "construction architect" in prompt_lower) and \
               ("software architect" not in prompt_lower and "software" not in prompt_lower and "code" not in prompt_lower):
                logger.warning("Misclassification detected: 'building architect' prompt likely triggered Code_Architect. Removing it.")
                adjusted_sequence.remove("Code_Architect")
                if "Generalist_Assistant" not in adjusted_sequence:
                    self._insert_persona_before_arbitrator(adjusted_sequence, "Generalist_Assistant")
        
        return adjusted_sequence

    def _insert_persona_before_arbitrator(self, sequence: List[str], persona: str):
        """Insert persona before the Impartial_Arbitrator in the sequence if not already present."""
        if persona in sequence:
            return
        
        arbitrator_index = len(sequence)
        if 'Impartial_Arbitrator' in sequence:
            arbitrator_index = sequence.index('Impartial_Arbitrator')
        
        sequence.insert(arbitrator_index, persona)
        logger.debug(f"Inserted persona '{persona}' before Arbitrator at index {arbitrator_index}.")

    def determine_persona_sequence(self, prompt: str, 
                                 domain: str, # Added domain as a required argument
                                 intermediate_results: Optional[Dict[str, Any]] = None,
                                 context_analysis_results: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Determine the optimal sequence of personas for processing the prompt.
        Dynamically adjusts the sequence based on intermediate results and context analysis.
        """
        prompt_lower = prompt.lower()
        
        if self.is_self_analysis_prompt(prompt):
            logger.info("Detected self-analysis prompt. Using standardized specialized persona sequence.")
            base_sequence = SELF_ANALYSIS_PERSONA_SEQUENCE
        else:
            # Use the persona_sets directly from personas.yaml as the base sequence
            if domain not in self.persona_sets:
                logger.warning(f"Domain '{domain}' not found in persona_sets. Falling back to 'General' sequence.")
                domain = "General" # Fallback to General if selected domain is not found
            
            # Retrieve the base sequence directly from the loaded persona_sets
            base_sequence = self.persona_sets.get(domain, [])
            
            if not base_sequence: # Fallback if the domain's sequence is empty or not found
                logger.warning(f"Persona set for domain '{domain}' is empty or invalid. Falling back to default 'General' sequence.")
                base_sequence = self.persona_sets.get("General", [])
                if not base_sequence: # Absolute fallback if 'General' is also empty
                    logger.error("No valid persona sequence found for 'General' domain. Using minimal fallback.")
                    base_sequence = ["Visionary_Generator", "Skeptical_Generator", "Impartial_Arbitrator"]

        # Ensure all personas in base_sequence actually exist in all_personas
        base_sequence = [p for p in base_sequence if p in self.all_personas]

        # Apply dynamic adjustments based on context analysis and intermediate results
        final_sequence = self._apply_dynamic_adjustment(base_sequence, intermediate_results, prompt_lower)
        
        # Further adjustments based on context analysis results (e.g., presence of test files)
        if context_analysis_results:
            relevant_files = context_analysis_results.get("relevant_files", [])
            test_file_count = sum(1 for file_path, _ in relevant_files if file_path.startswith('tests/'))
            code_file_count = sum(1 for file_path, _ in relevant_files if file_path.endswith(('.py', '.js', '.ts', '.java', '.go')))
            
            if test_file_count > 3 and "Test_Engineer" not in final_sequence:
                self._insert_persona_before_arbitrator(final_sequence, "Test_Engineer")
            
            if code_file_count > 5:
                if "Code_Architect" not in final_sequence:
                    self._insert_persona_before_arbitrator(final_sequence, "Code_Architect")
                if "Security_Auditor" not in final_sequence:
                    self._insert_persona_before_arbitrator(final_sequence, "Security_Auditor")
        
        # Ensure uniqueness and order
        seen = set()
        unique_sequence = []
        for persona in final_sequence:
            if persona not in seen:
                unique_sequence.append(persona)
                seen.add(persona)
        
        return unique_sequence