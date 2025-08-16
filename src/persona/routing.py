# src/persona/routing.py
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
    
    def is_self_analysis_prompt(self, prompt: str) -> bool:
       """Standardized method to detect self-analysis prompts using central constants"""
       return is_self_analysis_prompt(prompt)

    def _apply_dynamic_adjustment(self, sequence: List[str], intermediate_results: Optional[Dict[str, Any]], prompt_lower: str) -> List[str]:
        """Apply dynamic adjustments to persona sequence based on intermediate results quality metrics."""
        if not intermediate_results:
            intermediate_results = {}
        
        quality_metrics = {}
        # Extract quality metrics from intermediate results
        for step_name, result in intermediate_results.items():
            if isinstance(result, dict):
                # Check for quality_metrics directly in the result dictionary
                if 'quality_metrics' in result and isinstance(result['quality_metrics'], dict):
                    for metric_name, value in result['quality_metrics'].items():
                        quality_metrics[metric_name] = max(quality_metrics.get(metric_name, 0.0), value)
                # Also check for quality_metrics nested within persona output structures
                elif step_name.endswith("_Output") and isinstance(result, dict):
                    if "quality_metrics" in result and isinstance(result["quality_metrics"], dict):
                        for metric_name, value in result["quality_metrics"].items():
                            quality_metrics[metric_name] = max(quality_metrics.get(metric_name, 0.0), value)

        adjusted_sequence = sequence.copy()
        
        # --- Dynamic Adjustments based on Quality Metrics ---
        # These are heuristics and can be tuned.
        # Prioritize Security_Auditor if security risk is high
        if quality_metrics.get('security_risk_score', 0.0) > 0.7:
            self._insert_persona_before_arbitrator(adjusted_sequence, "Security_Auditor")
            logger.info("Prioritized Security_Auditor due to high security risk score.")
        
        # Prioritize Test_Engineer if test coverage estimate is low
        if quality_metrics.get('test_coverage_estimate', 1.0) < 0.5:
            self._insert_persona_before_arbitrator(adjusted_sequence, "Test_Engineer")
            logger.info("Prioritized Test_Engineer due to low test coverage estimate.")
        
        # Prioritize Code_Architect if maintainability or code quality is low
        if quality_metrics.get('maintainability_index', 1.0) < 0.7 or quality_metrics.get('code_quality', 1.0) < 0.7:
            self._insert_persona_before_arbitrator(adjusted_sequence, "Code_Architect")
            logger.info("Prioritized Code_Architect due to low maintainability or code quality.")

        # --- Dynamic Adjustments based on Prompt Misclassification ---
        # Check for common misclassifications and correct the sequence
        if "Code_Architect" in adjusted_sequence:
            # If prompt mentions building architecture but not software architecture
            if ("building architect" in prompt_lower or "construction architect" in prompt_lower) and \
               not ("software architect" in prompt_lower or "software" in prompt_lower or "code" in prompt_lower):
                logger.warning("Misclassification detected: 'building architect' prompt likely triggered Code_Architect. Removing it.")
                adjusted_sequence.remove("Code_Architect")
                # If removed, ensure a general assistant is present if not already
                if "Generalist_Assistant" not in adjusted_sequence and "Generalist_Assistant" in self.all_personas:
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
        Dynamically adjusts the sequence based on prompt keywords, domain,
        intermediate results, and context analysis.
        """
        prompt_lower = prompt.lower()
        
        # --- LLM SUGGESTION 2: Dynamic Persona Sequence for Self-Analysis ---
        # Check if it's a self-analysis prompt and apply specific sequences.
        if self.is_self_analysis_prompt(prompt):
            logger.info("Detected self-analysis prompt. Applying dynamic persona sequence.")
            
            # Start with a core self-analysis sequence
            base_sequence = ["Code_Architect", "Test_Engineer", "Constructive_Critic", "Impartial_Arbitrator", "Devils_Advocate"]
            
            # Dynamic adaptation based on specific self-analysis keywords
            # Prioritize Security_Auditor if security keywords are present
            if any(kw in prompt_lower for kw in ["security", "vulnerability", "exploit", "authentication", "threat", "risk"]):
                self._insert_persona_before_arbitrator(base_sequence, "Security_Auditor")
                logger.info("Self-analysis prompt is security-focused. Added Security_Auditor.")
            
            # Prioritize DevOps_Engineer if performance/DevOps keywords are present
            if any(kw in prompt_lower for kw in ["performance", "efficiency", "scalability", "devops", "ci/cd", "deployment"]):
                self._insert_persona_before_arbitrator(base_sequence, "DevOps_Engineer")
                logger.info("Self-analysis prompt is performance/DevOps-focused. Added DevOps_Engineer.")

            # Prioritize Code_Architect if maintainability/structure keywords are present
            if any(kw in prompt_lower for kw in ["maintainability", "readability", "structure", "refactor", "clean code"]):
                # If Code_Architect is already in the sequence, move it to the front
                if "Code_Architect" in base_sequence:
                    base_sequence.remove("Code_Architect")
                    base_sequence.insert(0, "Code_Architect") # Prioritize it at the beginning
                else:
                    # If not present, insert it early
                    self._insert_persona_before_arbitrator(base_sequence, "Code_Architect")
                logger.info("Self-analysis prompt is maintainability/structure-focused. Prioritized Code_Architect.")

            # Ensure Impartial_Arbitrator is always last for synthesis
            if "Impartial_Arbitrator" in base_sequence:
                base_sequence.remove("Impartial_Arbitrator")
            base_sequence.append("Impartial_Arbitrator")

            # Ensure Devils_Advocate is before Arbitrator but after critics
            if "Devils_Advocate" in base_sequence:
                base_sequence.remove("Devils_Advocate")
            
            # Find the index of the last critic or the Arbitrator if no critic exists
            insert_pos_for_advocate = len(base_sequence)
            if "Impartial_Arbitrator" in base_sequence:
                insert_pos_for_advocate = base_sequence.index("Impartial_Arbitrator")
            
            # Try to insert after Constructive_Critic if it exists
            if "Constructive_Critic" in base_sequence and base_sequence.index("Constructive_Critic") < insert_pos_for_advocate:
                critic_idx = base_sequence.index("Constructive_Critic")
                base_sequence.insert(critic_idx + 1, "Devils_Advocate")
            else:
                # Otherwise, insert it before the Arbitrator
                base_sequence.insert(insert_pos_for_advocate, "Devils_Advocate")
            
            final_sequence = base_sequence # Start with the dynamically built base sequence
            logger.info(f"Self-analysis persona sequence: {final_sequence}")

        else:
        # --- END LLM SUGGESTION 2 ---
            # Use the persona_sets directly from personas.yaml as the base sequence for non-self-analysis prompts
            if domain not in self.persona_sets:
                logger.warning(f"Domain '{domain}' not found in persona_sets. Falling back to 'General' sequence.")
                domain = "General" # Fallback to General if selected domain is not found
            
            base_sequence = self.persona_sets.get(domain, [])
            
            if not base_sequence: # Fallback if the domain's sequence is empty or not found
                logger.warning(f"Persona set for domain '{domain}' is empty or invalid. Falling back to default 'General' sequence.")
                base_sequence = self.persona_sets.get("General", [])
                if not base_sequence: # Absolute fallback if 'General' is also empty
                    logger.error("No valid persona sequence found for 'General' domain. Using minimal fallback.")
                    base_sequence = ["Visionary_Generator", "Skeptical_Generator", "Impartial_Arbitrator"]

            final_sequence = base_sequence.copy() # Start with the domain's base sequence

        # Apply dynamic adjustments based on context analysis and intermediate results
        # This will now apply to both self-analysis and general sequences
        final_sequence = self._apply_dynamic_adjustment(final_sequence, intermediate_results, prompt_lower)
        
        # Further adjustments based on context analysis results (e.g., presence of test files)
        if context_analysis_results:
            relevant_files = context_analysis_results.get("relevant_files", [])
            test_file_count = sum(1 for file_path, _ in relevant_files if file_path.startswith('tests/'))
            code_file_count = sum(1 for file_path, _ in relevant_files if file_path.endswith(('.py', '.js', '.ts', '.java', '.go')))
            
            # Insert Test_Engineer if many test files are relevant and it's not already in sequence
            if test_file_count > 3 and "Test_Engineer" not in final_sequence:
                self._insert_persona_before_arbitrator(final_sequence, "Test_Engineer")
            
            # Insert Code_Architect and Security_Auditor if many code files are relevant and they are not in sequence
            if code_file_count > 5:
                if "Code_Architect" not in final_sequence:
                    self._insert_persona_before_arbitrator(final_sequence, "Code_Architect")
                if "Security_Auditor" not in final_sequence:
                    self._insert_persona_before_arbitrator(final_sequence, "Security_Auditor")
        
        # Ensure uniqueness and order by removing duplicates while preserving order
        seen = set()
        unique_sequence = []
        for persona in final_sequence:
            if persona not in seen:
                unique_sequence.append(persona)
                seen.add(persona)
        
        return unique_sequence