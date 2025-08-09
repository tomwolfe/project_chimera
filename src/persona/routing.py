# src/persona/routing.py
"""
Dynamic persona routing system that selects appropriate personas
based on prompt analysis and intermediate results.
"""

from typing import List, Dict, Set, Optional, Any
import re
import json
from pathlib import Path
import logging # Ensure logging is imported
from functools import lru_cache # Import lru_cache

from src.models import PersonaConfig
# NEW: Import SELF_ANALYSIS_KEYWORDS and SELF_ANALYSIS_PERSONA_SEQUENCE
from src.constants import SELF_ANALYSIS_KEYWORDS, SELF_ANALYSIS_PERSONA_SEQUENCE # Import standardized sequence

logger = logging.getLogger(__name__) # Initialize logger

class PersonaRouter:
    """Determines the optimal sequence of personas for a given prompt."""
    
    def __init__(self, all_personas: Dict[str, PersonaConfig]):
        self.all_personas = all_personas
        
        # Updated DOMAIN_KEYWORDS with positive and negative keyword sets
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
                "positive": ["science", "research", "experiment", "data", "model", "hypothesis", 
                             "biology", "physics", "chemistry", "astronomy", "engineering", 
                             "algorithm", "computation", "genetics", "ecology", "neuroscience"],
                "negative": []
            },
            "business": {
                "positive": ["business", "market", "strategy", "finance", "investment", 
                             "startup", "profit", "revenue", "marketing", "sales", "operations", 
                             "management", "economy", "entrepreneurship", "product", "customer"],
                "negative": []
            },
            "creative": {
                "positive": ["creative", "art", "story", "design", "narrative", "fiction", 
                             "poetry", "music", "film", "painting", "sculpture", "writing", 
                             "imagination", "concept", "aesthetic"],
                "negative": []
            }
        }
        
        # Keywords to trigger specific personas based on intermediate results
        self.trigger_keywords = {
            "Security_Auditor": ["vulnerab", "security", "exploit", "hack", "auth", "encrypt", "threat", "risk", "malware", "penetration", "compliance", "attack vector", "data breach"],
            "Test_Engineer": ["test", "bug", "error", "quality", "coverage", "unit", "integration", "qa", "defect", "debug", "validate", "assertion", "failure", "edge case"],
            "DevOps_Engineer": ["deploy", "ci/cd", "pipeline", "infra", "monitor", "cloud", "docker", "k8s", "ops", "server", "automation", "release", "scalability", "reliability", "performance", "logging", "alerting"],
            "Code_Architect": ["architect", "design", "pattern", "scalab", "perform", "modular", "refactor", "system", "structure", "database", "api", "framework", "codebase", "maintainability", "technical debt", "separation of concerns"],
            "Constructive_Critic": ["improve", "refine", "optimize", "recommend", "suggest", "enhanc", "fix", "best practice"], # General improvement keywords
            "Skeptical_Generator": ["risk", "flaw", "limitation", "vulnerab", "bottleneck", "edge case", "failure point", "concern", "doubt"] # Keywords indicating skepticism
        }
    
    # --- NEW METHOD TO BE ADDED ---
    def determine_domain(self, prompt: str) -> str:
        """Determine the most appropriate domain for the given prompt.
        
        Returns the single most relevant domain based on keyword matching.
        """
        # First check if this is a self-analysis prompt using the class method
        if self.is_self_analysis_prompt(prompt):
            return "Software Engineering"
        
        prompt_lower = prompt.lower()
        best_match = "General"  # Default domain
        highest_score = 0
        
        # Define domain keywords for this specific method
        domain_keywords = {
            "General": {
                "keywords": ["analyze", "explain", "discuss", "consider", "think about"],
                "negative_keywords": []
            },
            "Science": {
                "keywords": ["scientific", "experiment", "hypothesis", "research", "data"],
                "negative_keywords": []
            },
            "Business": {
                "keywords": ["market", "business", "financial", "economy", "strategy"],
                "negative_keywords": []
            },
            "Creative": {
                "keywords": ["creative", "write", "story", "poem", "artistic"],
                "negative_keywords": []
            },
            "Software Engineering": {
                "keywords": ["code", "program", "software", "developer", "algorithm", "debug", "architecture", "engineering"],
                "negative_keywords": ["building", "construction"]  # Avoid confusion with building architecture
            }
        }
        
        for domain, config in domain_keywords.items():
            keywords = config["keywords"]
            negative_keywords = config["negative_keywords"]
            
            # Skip if any negative keyword is present
            if any(neg_kw in prompt_lower for neg_kw in negative_keywords):
                continue
                
            # If no negative keywords matched, check for positive keywords.
            score = sum(1 for kw in keywords if kw in prompt_lower)
            if score > highest_score:
                highest_score = score
                best_match = domain
        
        return best_match
    # --- END OF NEW METHOD ---

    def _analyze_prompt_domain(self, prompt: str) -> Set[str]:
        """
        Analyze prompt to determine relevant domains, using negative keyword filtering
        to prevent misclassifications.
        """
        prompt_lower = prompt.lower()
        matched_domains = set()
        
        for domain, config in self.domain_keywords.items():
            # Check for negative keywords first. If any are present, skip this domain.
            has_negative_match = any(keyword in prompt_lower for keyword in config.get("negative", []))
            if has_negative_match:
                continue # Skip this domain if a negative keyword is found
                
            # If no negative keywords matched, check for positive keywords.
            has_positive_match = any(keyword in prompt_lower for keyword in config.get("positive", []))
            if has_positive_match:
                matched_domains.add(domain)
        
        # Fallback to "General" if no specific domains are matched after filtering.
        return matched_domains if matched_domains else {"General"}
    
    def _get_domain_specific_personas(self, domains: Set[str]) -> List[str]:
        """Get personas relevant to the detected domains."""
        # Define which personas are relevant for each domain
        domain_persona_map = {
            "security": ["Security_Auditor"],
            "architecture": ["Code_Architect"],
            "testing": ["Test_Engineer"],
            "devops": ["DevOps_Engineer"],
            "scientific": ["Scientific_Visionary", "Scientific_Analyst"],
            "business": ["Business_Innovator", "Business_Strategist"],
            "creative": ["Creative_Visionary", "Creative_Thinker"],
            "general": [] # General domain doesn't add specific personas beyond core ones
        }
        
        # Collect all relevant personas, ensuring uniqueness and including core personas
        relevant_personas = set()
        
        # Add core personas that are always useful
        core_personas = ["Visionary_Generator", "Skeptical_Generator", "Constructive_Critic", "Impartial_Arbitrator", "Devils_Advocate"]
        relevant_personas.update(core_personas)
        
        # Add domain-specific personas
        for domain in domains:
            if domain in domain_persona_map:
                relevant_personas.update(domain_persona_map[domain])
        
        # Ensure all personas exist in the provided all_personas dictionary
        # This is a safeguard; ideally, the persona sets in personas.yaml are consistent.
        valid_personas = {p for p in relevant_personas if p in self.all_personas}
        
        # Return a list, maintaining a sensible default order
        # Core personas first, then domain-specific ones.
        ordered_sequence = []
        for p in core_personas:
            if p in valid_personas:
                ordered_sequence.append(p)
                valid_personas.remove(p) # Remove to avoid duplicates
        
        # Add remaining domain-specific personas, sorted alphabetically for consistency
        ordered_sequence.extend(sorted(list(valid_personas)))
        
        return ordered_sequence
    
    # --- NEW METHOD: is_self_analysis_prompt ---
    def is_self_analysis_prompt(self, prompt: str) -> bool:
       """Standardized method to detect self-analysis prompts using central constants"""
       # Use the imported function from constants.py
       from src.constants import is_self_analysis_prompt # Import here for clarity
       return is_self_analysis_prompt(prompt)
       # The actual check is done via `from src.constants import is_self_analysis_prompt` in core.py
       # and then called as `is_self_analysis_prompt(self.initial_prompt)`
       # This method here is just a placeholder to satisfy the structure if it were called directly.
       # The correct implementation is in core.py using the imported function.
       # If this method were to be used, it would look like:
       # from src.constants import is_self_analysis_prompt
       # return is_self_analysis_prompt(prompt)
       pass # This method should not be called directly, rely on import in core.py

    # --- MODIFIED HELPER METHOD: _apply_dynamic_adjustment ---
    def _apply_dynamic_adjustment(self, sequence: List[str], intermediate_results: Optional[Dict[str, Any]], prompt_lower: str) -> List[str]:
        """Apply dynamic adjustments to persona sequence based on intermediate results quality metrics."""
        # Handle case where intermediate_results is None (can happen during initial self-analysis)
        if not intermediate_results:
            intermediate_results = {}
        
        # Extract quality metrics from intermediate results
        # This part needs to correctly parse the quality_metrics from the results.
        # The structure might be nested, e.g., result['quality_metrics'].
        quality_metrics = {}
        for step_name, result in intermediate_results.items():
            # Look for results that contain quality metrics, often from specific personas
            if isinstance(result, dict):
                if 'quality_metrics' in result and isinstance(result['quality_metrics'], dict):
                    # Merge metrics, prioritizing higher scores if multiple personas report on the same metric
                    for metric_name, value in result['quality_metrics'].items():
                        quality_metrics[metric_name] = max(quality_metrics.get(metric_name, 0.0), value)
                # Also check for specific persona outputs that might contain metrics directly
                elif step_name.endswith("_Output") and isinstance(result, str):
                    # Simple keyword check for metrics within string outputs (less reliable)
                    if "code_quality_score" in result: quality_metrics["code_quality"] = max(quality_metrics.get("code_quality", 0.0), 0.6) # Example heuristic
                    if "security_risk" in result: quality_metrics["security_risk_score"] = max(quality_metrics.get("security_risk_score", 0.0), 0.7)
                elif step_name.endswith("_Output") and isinstance(result, dict): # Handle if output is a dict itself
                    if "quality_metrics" in result and isinstance(result["quality_metrics"], dict):
                        for metric_name, value in result["quality_metrics"].items():
                            quality_metrics[metric_name] = max(quality_metrics.get(metric_name, 0.0), value)

        adjusted_sequence = sequence.copy()
        
        # --- Priority adjustments based on detected needs ---
        
        # If code quality is low, prioritize code-focused personas
        if quality_metrics.get('code_quality', 1.0) < 0.7:
            self._insert_persona_before_arbitrator(adjusted_sequence, 'Code_Architect')
            self._insert_persona_before_arbitrator(adjusted_sequence, 'Security_Auditor') # Security is often tied to code quality
            self._insert_persona_before_arbitrator(adjusted_sequence, 'Test_Engineer') # Testing is key to quality
        
        # If reasoning depth is low, prioritize personas that can add depth
        if quality_metrics.get('reasoning_depth', 1.0) < 0.6:
            # Devils_Advocate can challenge and deepen the reasoning
            self._insert_persona_before_arbitrator(adjusted_sequence, 'Devils_Advocate')
            # Constructive_Critic can also help refine arguments
            self._insert_persona_before_arbitrator(adjusted_sequence, 'Constructive_Critic')
        
        # If test coverage is low, prioritize testing personas
        if quality_metrics.get('test_coverage_estimate', 1.0) < 0.5:
            self._insert_persona_before_arbitrator(adjusted_sequence, 'Test_Engineer')
        
        # If security risk is high, prioritize security personas
        if quality_metrics.get('security_risk_score', 0.0) > 0.7:
            self._insert_persona_before_arbitrator(adjusted_sequence, 'Security_Auditor')
            # Also consider DevOps for security operations
            self._insert_persona_before_arbitrator(adjusted_sequence, 'DevOps_Engineer')

        # Remove redundant personas if certain quality thresholds are met
        # Example: If architectural coherence is very high, Code_Architect might be less critical
        if quality_metrics.get('architectural_coherence', 1.0) > 0.9 and 'Code_Architect' in adjusted_sequence:
            logger.debug("Removing Code_Architect due to high architectural coherence.")
            adjusted_sequence.remove('Code_Architect')
        
        # Example: If the prompt was simple and reasoning depth is high, maybe remove Devils_Advocate
        if quality_metrics.get('reasoning_depth', 1.0) > 0.8 and len(prompt_lower.split()) < 50 and 'Devils_Advocate' in adjusted_sequence:
            logger.debug("Removing Devils_Advocate for a simple prompt with high reasoning depth.")
            adjusted_sequence.remove('Devils_Advocate')

        # --- Minimal 80/20 Refinement for Framework Selection ---
        # Add a simple check to prevent common misclassifications based on prompt context.
        # This avoids modifying personas.yaml or adding complex scoring.
        
        # Example: Prevent "building architect" from triggering Code_Architect
        if "Code_Architect" in adjusted_sequence:
            if ("building architect" in prompt_lower or "construction architect" in prompt_lower) and \
               ("software architect" not in prompt_lower and "software" not in prompt_lower and "code" not in prompt_lower):
                
                logger.warning("Misclassification detected: 'building architect' prompt likely triggered Code_Architect. Removing it.")
                adjusted_sequence.remove("Code_Architect")
                # Optionally, add a more general persona if a specific one is removed
                if "Generalist_Assistant" not in adjusted_sequence:
                    self._insert_persona_before_arbitrator(adjusted_sequence, "Generalist_Assistant")
        
        # Add similar checks for other known problematic keyword overlaps if identified.
        # --- End of Refinement ---

        # Ensure the final sequence is unique and maintains a logical order.
        # This step is important if multiple triggers add the same persona or if
        # the insertion logic creates duplicates.
        seen = set()
        unique_sequence = []
        for persona in adjusted_sequence:
            if persona not in seen:
                unique_sequence.append(persona)
                seen.add(persona)
        
        return unique_sequence

    def _insert_persona_before_arbitrator(self, sequence: List[str], persona: str):
        """Insert persona before the Impartial_Arbitrator in the sequence if not already present."""
        if persona in sequence:
            return  # Already in sequence
        
        # Find the index of the Arbitrator, or append if not found
        arbitrator_index = len(sequence)
        if 'Impartial_Arbitrator' in sequence:
            arbitrator_index = sequence.index('Impartial_Arbitrator')
        
        # Insert the persona at the determined index
        sequence.insert(arbitrator_index, persona)
        logger.debug(f"Inserted persona '{persona}' before Arbitrator at index {arbitrator_index}.")


    def determine_persona_sequence(self, prompt: str, 
                                 intermediate_results: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Determine the optimal sequence of personas for processing the prompt.
        Dynamically adjusts the sequence based on intermediate results.
        
        Returns a list of persona names in execution order.
        """
        
        prompt_lower = prompt.lower()
        
        # 1. Primary check: Is this a self-analysis prompt? Use the centralized method.
        # Ensure the import is correct and the function is called properly.
        from src.constants import is_self_analysis_prompt # Import here for clarity
        if is_self_analysis_prompt(prompt):
            logger.info("Detected self-analysis prompt. Using standardized specialized persona sequence.")
            # Start with the standardized, comprehensive self-analysis sequence
            base_sequence = SELF_ANALYSIS_PERSONA_SEQUENCE
            
            # Apply dynamic adjustment if intermediate results exist
            return self._apply_dynamic_adjustment(base_sequence, intermediate_results, prompt_lower)
        
        # --- Domain-specific routing continues below (original logic) ---
        # 2. Initial domain-based sequence
        domains = self._analyze_prompt_domain(prompt)
        base_sequence = self._get_domain_specific_personas(domains)
        
        # Get core personas and domain experts
        core_order = ["Visionary_Generator", "Skeptical_Generator"]
        domain_experts = [p for p in base_sequence
                         if p not in core_order and p != "Impartial_Arbitrator"]
        
        final_sequence = core_order + domain_experts
        if "Impartial_Arbitrator" in base_sequence: # Ensure Arbitrator is included if it was in the base set
            final_sequence.append("Impartial_Arbitrator")
        
        # 3. Apply dynamic adjustment based on intermediate results (using the helper method)
        return self._apply_dynamic_adjustment(final_sequence, intermediate_results, prompt_lower)