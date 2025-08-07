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

from src.models import PersonaConfig

# NEW: Import constants for self-analysis keywords
from src.constants import SELF_ANALYSIS_KEYWORDS

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
    
    # --- REFACTORED METHOD ---
    def determine_persona_sequence(self, prompt: str, 
                                 intermediate_results: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Determine optimal persona sequence with clear decision hierarchy and centralized logic.
        """
        prompt_lower = prompt.lower()
        
        # 1. Primary check: Is this a self-analysis prompt?
        if self._is_self_analysis_prompt(prompt_lower):
            logger.info("Detected self-analysis prompt. Using specialized persona sequence.")
            return self._get_self_analysis_sequence()
        
        # 2. Secondary check: Domain-specific routing
        domains = self._analyze_prompt_domain(prompt)
        base_sequence = self._get_domain_specific_personas(domains)
        
        # 3. Apply core framework with domain experts
        core_order = ["Visionary_Generator", "Skeptical_Generator"]
        domain_experts = [p for p in base_sequence
                         if p not in core_order and p != "Impartial_Arbitrator"]
        
        # Ensure Arbitrator is included if it was part of the base sequence
        final_sequence = core_order + domain_experts
        if "Impartial_Arbitrator" in base_sequence and "Impartial_Arbitrator" not in final_sequence:
            final_sequence.append("Impartial_Arbitrator")
        
        # 4. Apply final sequence safeguards (e.g., misclassification fixes)
        final_sequence = self._apply_sequence_safeguards(final_sequence, prompt_lower)
        
        # 5. Ensure uniqueness and order
        return self._ensure_unique_and_ordered_sequence(final_sequence)

    # --- NEW HELPER METHODS (extracted from original logic) ---
    def _is_self_analysis_prompt(self, prompt_lower: str) -> bool:
        """Checks if the prompt contains keywords indicating self-analysis."""
        # Import from constants for consistency (as per Suggestion 1)
        return any(keyword in prompt_lower for keyword in SELF_ANALYSIS_KEYWORDS)

    def _get_self_analysis_sequence(self) -> List[str]:
        """Returns the specialized persona sequence for self-analysis tasks."""
        # This sequence prioritizes code-focused roles.
        return [
            "Code_Architect",
            "Skeptical_Generator",
            "Constructive_Critic",
            "Test_Engineer",
            "Impartial_Arbitrator",
            "Devils_Advocate"
        ]

    def _apply_sequence_safeguards(self, sequence: List[str], prompt_lower: str) -> List[str]:
        """Applies final validation and adjustments to the persona sequence."""
        # Prevent misclassification: e.g., "building architect" vs "software architect"
        if "Code_Architect" in sequence:
            building_terms = ["building architect", "construction architect"]
            software_terms = ["software architect", "software"]
            
            if any(term in prompt_lower for term in building_terms) and \
               not any(term in prompt_lower for term in software_terms):
                
                logger.warning("Misclassification detected: 'building architect' prompt likely triggered Code_Architect. Removing it.")
                sequence.remove("Code_Architect")
                # Optionally add a general fallback if a specific persona is removed
                if "Generalist_Assistant" not in sequence:
                    sequence.append("Generalist_Assistant")
        
        # Add other safeguard checks here as needed.
        return sequence

    def _ensure_unique_and_ordered_sequence(self, sequence: List[str]) -> List[str]:
        """Ensures the final sequence contains unique persona names in a logical order."""
        seen = set()
        unique_sequence = []
        for persona in sequence:
            if persona not in seen:
                unique_sequence.append(persona)
                seen.add(persona)
        return unique_sequence