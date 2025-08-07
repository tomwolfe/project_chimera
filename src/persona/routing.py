# src/persona/routing.py
"""
Dynamic persona routing system that selects appropriate personas
based on prompt analysis and intermediate results.
"""

# FIX: Added 'Any' to the import statement from 'typing'
from typing import List, Dict, Set, Optional, Any
import re
from src.models import PersonaConfig, ReasoningFrameworkConfig # Assuming these models exist

class PersonaRouter:
    """Determines the optimal sequence of personas for a given prompt."""
    
    def __init__(self, all_personas: Dict[str, PersonaConfig]):
        self.all_personas = all_personas
        self.domain_keywords = {
            "security": ["security", "vulnerab", "auth", "encrypt", "hack", "threat", "risk"],
            "architecture": ["architect", "design", "pattern", "scalab", "perform", "modular"],
            "testing": ["test", "cover", "unit", "integration", "bug", "error", "quality"],
            "devops": ["deploy", "ci/cd", "pipeline", "infra", "monitor", "cloud", "docker", "kubernetes"],
            "scientific": ["science", "research", "experiment", "data", "model", "hypothesis"]
        }
    
    def _analyze_prompt_domain(self, prompt: str) -> Set[str]:
        """Identify relevant domains in the prompt."""
        prompt_lower = prompt.lower()
        detected_domains = set()
        
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if re.search(rf'\b{keyword}', prompt_lower):
                    detected_domains.add(domain)
                    break
        
        # Default to general if no domains detected
        return detected_domains if detected_domains else {"general"}
    
    def _get_domain_specific_personas(self, domains: Set[str]) -> List[str]:
        """Get personas relevant to the detected domains."""
        domain_personas = {
            "security": ["Security_Auditor", "Devils_Advocate"],
            "architecture": ["Code_Architect", "Constructive_Critic"],
            "testing": ["Test_Engineer", "Constructive_Critic"],
            "devops": ["DevOps_Engineer"],
            "scientific": ["Scientific_Visionary", "Skeptical_Generator"]
        }
        
        # Collect all relevant personas
        relevant_personas = set()
        for domain in domains:
            if domain in domain_personas:
                relevant_personas.update(domain_personas[domain])
        
        # Always include core personas
        relevant_personas.update(["Visionary_Generator", "Skeptical_Generator", "Impartial_Arbitrator"])
        
        return list(relevant_personas)
    
    def _optimize_persona_sequence(self, initial_personas: List[str], 
                                 intermediate_results: Dict[str, Any]) -> List[str]:
        """Optimize the persona sequence based on intermediate results."""
        # If certain concerns were already addressed, skip redundant personas
        resolved_concerns = set()
        for step_name, result in intermediate_results.items():
            if "security" in step_name.lower() or "vulnerab" in str(result).lower():
                resolved_concerns.add("security")
            if "performance" in step_name.lower() or "bottleneck" in str(result).lower():
                resolved_concerns.add("performance")
        
        # Filter out personas that would address already-resolved concerns
        optimized_personas = []
        for persona in initial_personas:
            if persona == "Security_Auditor" and "security" in resolved_concerns:
                continue
            if persona == "Code_Architect" and "performance" in resolved_concerns:
                continue
            optimized_personas.append(persona)
        
        return optimized_personas
    
    def determine_persona_sequence(self, prompt: str, 
                                 intermediate_results: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Determine the optimal sequence of personas for processing the prompt.
        
        Returns a list of persona names in execution order.
        """
        # Analyze prompt to determine relevant domains
        domains = self._analyze_prompt_domain(prompt)
        
        # Get relevant personas for detected domains
        relevant_personas = self._get_domain_specific_personas(domains)
        
        # Optimize sequence based on intermediate results if available
        if intermediate_results:
            return self._optimize_persona_sequence(relevant_personas, intermediate_results)
        
        # Default ordering: visionary -> skeptics -> domain experts -> synthesizer
        core_order = ["Visionary_Generator", "Skeptical_Generator"]
        domain_experts = [p for p in relevant_personas 
                         if p not in core_order and p != "Impartial_Arbitrator"]
        return core_order + domain_experts + ["Impartial_Arbitrator"]