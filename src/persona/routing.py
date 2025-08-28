# src/persona/routing.py
"""
Dynamic persona routing system that selects appropriate personas
based on prompt analysis and intermediate results.
"""

import numpy as np 
from sentence_transformers import SentenceTransformer 
from typing import List, Dict, Set, Optional, Any, Tuple
import re
import json
from pathlib import Path
import logging
from functools import lru_cache

from src.models import PersonaConfig
# REMOVED: from src.constants import SELF_ANALYSIS_KEYWORDS
from src.constants import SELF_ANALYSIS_PERSONA_SEQUENCE # Re-added specifically for fallback
# REMOVED: from src.constants import is_self_analysis_prompt 
from src.utils.prompt_analyzer import PromptAnalyzer # NEW IMPORT

logger = logging.getLogger(__name__)

class PersonaRouter:
    """Determines the optimal sequence of personas for a given prompt."""
    
    def __init__(self, all_personas: Dict[str, PersonaConfig], persona_sets: Dict[str, List[str]], prompt_analyzer: PromptAnalyzer): # Accept PromptAnalyzer
        self.all_personas = all_personas
        self.persona_sets = persona_sets 
        self.prompt_analyzer = prompt_analyzer # Store the PromptAnalyzer instance
        
        # Initialize SentenceTransformer for semantic routing
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.persona_embeddings = self._generate_persona_embeddings() 

        # REMOVED: self.domain_keywords (now in PromptAnalyzer)
        
        self.trigger_keywords = {
            "Security_Auditor": ["vulnerab", "security", "exploit", "hack", "auth", "encrypt", "threat", 
                                 "risk", "malware", "penetration", "compliance", "firewall", "ssl", "tls",
                                 "attack vector", "data breach"],
            "Test_Engineer": ["test", "cover", "unit", "integration", "bug", "error", "quality", 
                              "qa", "defect", "debug", "validate", "assertion", "failure", "edge case",
                              "pytest", "unittest"],
            "DevOps_Engineer": ["deploy", "ci/cd", "pipeline", "infra", "monitor", "cloud", "docker", 
                                "k8s", "ops", "server", "automation", "release", "scalability", 
                                "reliability", "performance", "logging", "alerting"],
            "Code_Architect": ["architect", "design", "pattern", "scalab", "perform", "modular", "refactor", 
                               "system", "structure", "database", "api", "framework", "codebase", 
                               "maintainability", "technical debt", "separation of concerns", 
                               "microservice", "monolith", "backend", "frontend"],
            "Constructive_Critic": ["improve", "refine", "optimize", "recommend", "suggest", "enhanc", "fix", "best practice"],
            "Skeptical_Generator": ["risk", "flaw", "limitation", "vulnerab", "bottleneck", "edge case", "failure point", "concern", "doubt"]
        }
    
    def _generate_persona_embeddings(self) -> Dict[str, Any]:
        """Generates embeddings for all persona descriptions for semantic routing."""
        embeddings = {}
        for name, config in self.all_personas.items():
            if config.description:
                embeddings[name] = self.model.encode([config.description])[0]
        return embeddings

    # REMOVED: is_self_analysis_prompt method, now delegated to self.prompt_analyzer

    def _should_include_test_engineer(self, prompt_lower: str, context_analysis_results: Optional[Dict[str, Any]]) -> bool:
        """Determine if Test_Engineer persona is needed based on prompt and context."""
        
        # Keywords indicating testing focus in the prompt
        testing_keywords = ["test", "unit test", "integration test", "e2e test", "test coverage", 
                            "bug", "fix bug", "debug", "qa", "quality assurance", "validate", 
                            "verify", "assertion", "test case", "test suite", "pytest", "unittest"]
        if any(keyword in prompt_lower for keyword in testing_keywords):
            return True
        
        # Check for test files in context analysis results
        if context_analysis_results and context_analysis_results.get("relevant_files"):
            for file_path, _ in context_analysis_results["relevant_files"]:
                if "test" in file_path.lower() or "spec" in file_path.lower() or file_path.startswith('tests/'):
                    return True
        
        return False

    def _apply_dynamic_adjustment(self, sequence: List[str], intermediate_results: Optional[Dict[str, Any]], prompt_lower: str, domain: str, context_analysis_results: Optional[Dict[str, Any]]) -> List[str]:
        """Apply dynamic adjustments to persona sequence based on intermediate results quality metrics."""
        if not intermediate_results:
            intermediate_results = {}
        
        # Extract and map quality metrics from Context_Aware_Assistant's output
        context_analysis_output = intermediate_results.get("Context_Aware_Assistant_Output")
        if context_analysis_output and isinstance(context_analysis_output, dict):
            key_modules = context_analysis_output.get('key_modules', [])
            security_concerns = context_analysis_output.get('security_concerns', [])
            
            avg_code_quality = 1.0 # Default to high
            avg_complexity = 0.0 # Default to low
            if key_modules:
                avg_code_quality = sum(m.get('code_quality_score', 1.0) for m in key_modules) / len(key_modules)
                avg_complexity = sum(m.get('complexity_score', 0.0) for m in key_modules) / len(key_modules)
            
            # Prioritize Security_Auditor if security concerns are high
            if security_concerns:
                self._insert_persona_before_arbitrator(sequence, "Security_Auditor") # Use sequence directly
                logger.info("Prioritized Security_Auditor due to security concerns from context analysis.")
            
            # Prioritize Code_Architect if maintainability/code quality is low or complexity is high
            if avg_code_quality < 0.7 or avg_complexity > 0.7:
                self._insert_persona_before_arbitrator(sequence, "Code_Architect") # Use sequence directly
                logger.info("Prioritized Code_Architect due to low code quality/maintainability or high complexity from context analysis.")

        adjusted_sequence = sequence.copy()
        
        # Enhanced misclassification detection for architecture terms
        if "Code_Architect" in adjusted_sequence:
            building_arch_terms = ["building", "construction", "structural", "physical", "blueprint", "skyscraper", "house", "design", "floor plan"]
            software_arch_terms = ["software", "code", "system", "api", "database", "backend", "frontend"]
            
            building_count = sum(1 for term in building_arch_terms if term in prompt_lower)
            software_count = sum(1 for term in software_arch_terms if term in prompt_lower)
            
            if building_count > software_count and building_count >= 2 and software_count == 0:
                logger.warning(f"Misclassification detected: Building architecture prompt likely triggered Code_Architect. Removing it.")
                adjusted_sequence.remove("Code_Architect")
                if "Creative_Thinker" not in adjusted_sequence and "Creative_Thinker" in self.all_personas:
                    self._insert_persona_before_arbitrator(adjusted_sequence, "Creative_Thinker")
        
        # --- Conditional inclusion/exclusion of Test_Engineer ---
        if domain == "Software Engineering":
            if "Test_Engineer" in adjusted_sequence and not self._should_include_test_engineer(prompt_lower, context_analysis_results):
                adjusted_sequence.remove("Test_Engineer")
                logger.info("Removed Test_Engineer from sequence as no testing context/keywords detected.")
            elif "Test_Engineer" not in adjusted_sequence and self._should_include_test_engineer(prompt_lower, context_analysis_results):
                # If Test_Engineer is not in the base sequence but is needed, insert it
                self._insert_persona_before_arbitrator(adjusted_sequence, "Test_Engineer")
                logger.info("Added Test_Engineer to sequence due to testing context/keywords detected.")

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
        if self.prompt_analyzer.is_self_analysis_prompt(prompt): # DELEGATE to PromptAnalyzer
            logger.info("Detected self-analysis prompt. Applying dynamic persona sequence from 'Self-Improvement' set.")
            
            # Use the defined 'Self-Improvement' persona set, with a fallback to the hardcoded sequence
            base_sequence = self.persona_sets.get("Self-Improvement", SELF_ANALYSIS_PERSONA_SEQUENCE).copy()
            
            # Dynamic adaptation for self-analysis based on specific keywords (existing logic)
            # Prioritize Security_Auditor if security keywords are present
            if any(kw in prompt_lower for kw in ["security", "vulnerability", "exploit", "authentication", "threat", "risk"]) and "Security_Auditor" not in base_sequence:
                self._insert_persona_before_arbitrator(base_sequence, "Security_Auditor")
                logger.info("Self-analysis prompt is security-focused. Added Security_Auditor.")
            
            # Prioritize DevOps_Engineer if performance/DevOps keywords are present
            if any(kw in prompt_lower for kw in ["performance", "efficiency", "scalability", "devops", "ci/cd", "deployment"]) and "DevOps_Engineer" not in base_sequence:
                self._insert_persona_before_arbitrator(base_sequence, "DevOps_Engineer")
                logger.info("Self-analysis prompt is performance/DevOps-focused. Added DevOps_Engineer.")

            # Prioritize Code_Architect if maintainability/structure keywords are present
            if any(kw in prompt_lower for kw in ["maintainability", "readability", "structure", "refactor", "clean code"]) and "Code_Architect" not in base_sequence:
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
            # Semantic similarity for initial persona selection (new)
            if self.persona_embeddings:
                prompt_embedding = self.model.encode([prompt])[0]
                semantic_scores = {}
                for p_name, p_embedding in self.persona_embeddings.items():
                    semantic_scores[p_name] = np.dot(prompt_embedding, p_embedding) / (np.linalg.norm(prompt_embedding) * np.linalg.norm(p_embedding))
                
                # Boost personas from the selected domain
                domain_personas = self.persona_sets.get(domain, [])
                for p_name in domain_personas:
                    semantic_scores[p_name] = semantic_scores.get(p_name, 0.0) + 0.2 # Small boost for domain relevance

                # Select top N personas based on semantic score
                top_semantic_personas = sorted(semantic_scores.items(), key=lambda x: x[1], reverse=True)[:5] # Top 5
                initial_semantic_sequence = [p[0] for p in top_semantic_personas if p[0] in self.all_personas]
                
                # Merge with the base sequence from persona_sets, prioritizing semantic matches
                base_sequence = []
                for p_name in initial_semantic_sequence:
                    if p_name not in base_sequence:
                        base_sequence.append(p_name)
                for p_name in self.persona_sets.get(domain, []):
                    if p_name not in base_sequence:
                        base_sequence.append(p_name)
                
                # Ensure synthesis persona is always present and last
                synthesis_persona = "Impartial_Arbitrator" if domain == "Software Engineering" else "General_Synthesizer"
                if synthesis_persona not in base_sequence:
                    base_sequence.append(synthesis_persona)
                else:
                    base_sequence.remove(synthesis_persona)
                    base_sequence.append(synthesis_persona)

                logger.info(f"Initial semantic-driven persona sequence: {base_sequence}")
            else:
                # Fallback to existing logic if semantic model fails
                if domain not in self.persona_sets:
                    logger.warning(f"Domain '{domain}' not found in persona_sets. Falling back to 'General' sequence.")
                    domain = "General"
                base_sequence = self.persona_sets.get(domain, [])
                if not base_sequence:
                    logger.error("No valid persona sequence found. Using minimal fallback.")
                    base_sequence = ["Visionary_Generator", "Skeptical_Generator", "Impartial_Arbitrator"]

            final_sequence = base_sequence.copy() # Start with the domain's base sequence

        # Apply dynamic adjustments based on context analysis and intermediate results
        # This will now apply to both self-analysis and general sequences
        final_sequence = self._apply_dynamic_adjustment(final_sequence, intermediate_results, prompt_lower, domain, context_analysis_results)
        
        # Further adjustments based on context analysis results (e.g., presence of test files)
        if context_analysis_results:
            relevant_files = context_analysis_results.get("relevant_files", [])
            test_file_count = sum(1 for file_path, _ in relevant_files if file_path.startswith('tests/'))
            code_file_count = sum(1 for file_path, _ in relevant_files if file_path.endswith(('.py', '.js', '.ts', '.java', '.go')))
            
            # Insert Test_Engineer if many test files are relevant and it's not already in sequence
            # This is now handled by _apply_dynamic_adjustment, but keeping as a fallback/double-check
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