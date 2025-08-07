# src/persona/routing.py
"""
Dynamic persona routing system that selects appropriate personas
based on prompt analysis and intermediate results.
"""

from typing import List, Dict, Set, Optional, Any
import re
import json # Added for processing intermediate results
from pathlib import Path # Added for potential future path-based routing

from src.models import PersonaConfig # Assuming these models exist

class PersonaRouter:
    """Determines the optimal sequence of personas for a given prompt."""
    
    def __init__(self, all_personas: Dict[str, PersonaConfig]):
        self.all_personas = all_personas
        
        # Keywords for initial domain detection
        self.domain_keywords = {
            "security": ["vulnerab", "security", "exploit", "hack", "auth", "encrypt", "threat", "risk", "malware", "penetration", "compliance"],
            "architecture": ["architect", "design", "pattern", "scalab", "perform", "modular", "refactor", "system", "structure", "database", "api", "framework"],
            "testing": ["test", "cover", "unit", "integration", "bug", "error", "quality", "qa", "defect", "debug", "validate"],
            "devops": ["deploy", "ci/cd", "pipeline", "infra", "monitor", "cloud", "docker", "k8s", "ops", "server", "automation", "release", "scalability", "reliability", "performance", "logging", "alerting"],
            "scientific": ["science", "research", "experiment", "data", "model", "hypothesis", "biology", "physics", "chemistry", "astronomy", "engineering", "algorithm", "computation"],
            "business": ["business", "market", "strategy", "finance", "investment", "startup", "profit", "revenue", "marketing", "sales", "operations", "management", "economy", "entrepreneurship"],
            "creative": ["creative", "art", "story", "design", "narrative", "fiction", "poetry", "music", "film", "painting", "sculpture", "writing", "imagination", "concept", "aesthetic"]
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
        """Identify relevant domains in the prompt."""
        prompt_lower = prompt.lower()
        detected_domains = set()
        
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                # Use word boundaries for more precise matching
                if re.search(rf'\b{keyword}\b', prompt_lower):
                    detected_domains.add(domain)
                    break # Found a keyword for this domain, move to the next domain
        
        # Default to general if no specific domains detected
        return detected_domains if detected_domains else {"general"}
    
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
    
    def determine_persona_sequence(self, prompt: str, 
                                 intermediate_results: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Determine the optimal sequence of personas for processing the prompt.
        Dynamically adjusts the sequence based on intermediate results.
        
        Returns a list of persona names in execution order.
        """
        # 1. Initial domain-based sequence
        domains = self._analyze_prompt_domain(prompt)
        base_sequence = self._get_domain_specific_personas(domains)
        
        # Ensure core personas are present and ordered correctly
        core_order = ["Visionary_Generator", "Skeptical_Generator"]
        # Get domain experts from the base sequence, excluding core ones and Arbitrator
        domain_experts = [p for p in base_sequence 
                         if p not in core_order and p != "Impartial_Arbitrator"]
        
        # Start with the core sequence, then domain experts, then the arbitrator
        final_sequence = core_order + domain_experts
        if "Impartial_Arbitrator" in base_sequence: # Ensure Arbitrator is included if it was in the base set
            final_sequence.append("Impartial_Arbitrator")
        
        # 2. Dynamic adjustment based on intermediate results
        if intermediate_results:
            # Analyze previous outputs for specific trigger keywords
            triggered_personas_to_add = set()
            
            # Iterate through all intermediate steps
            for step_name, result in intermediate_results.items():
                # Skip steps that are not actual persona outputs (e.g., token counts, errors)
                if not (step_name.endswith("_Output") or step_name.endswith("_Critique")):
                    continue

                # Convert result to string for keyword searching. Handle dicts by serializing.
                result_text = ""
                if isinstance(result, dict):
                    try:
                        result_text = json.dumps(result)
                    except TypeError: # Handle cases where result might not be JSON serializable
                        result_text = str(result)
                elif isinstance(result, str):
                    result_text = result
                else:
                    result_text = str(result) # Fallback for other types

                # Check for trigger keywords in the result text
                for persona_name, keywords in self.trigger_keywords.items():
                    # Only consider adding personas not already in the sequence
                    if persona_name not in final_sequence:
                        for keyword in keywords:
                            # Use word boundaries for more precise matching
                            if re.search(rf'\b{keyword}\b', result_text, re.IGNORECASE):
                                triggered_personas_to_add.add(persona_name)
                                break # Found a trigger for this persona, move to the next persona
            
            # If new personas were triggered, insert them into the sequence.
            # A good place is before the Impartial_Arbitrator, or based on their role.
            # For simplicity, we'll insert them before the Arbitrator.
            if triggered_personas_to_add:
                # Sort triggered personas alphabetically for deterministic order
                sorted_triggered_personas = sorted(list(triggered_personas_to_add))
                
                # Find the index where the Arbitrator is (or where it would be inserted)
                arbitrator_index = len(final_sequence)
                if "Impartial_Arbitrator" in final_sequence:
                    arbitrator_index = final_sequence.index("Impartial_Arbitrator")
                
                # Insert the triggered personas before the Arbitrator
                for persona_to_insert in sorted_triggered_personas:
                    if persona_to_insert not in final_sequence: # Double check uniqueness
                        final_sequence.insert(arbitrator_index, persona_to_insert)
                        arbitrator_index += 1 # Adjust index for subsequent insertions
        
        # Ensure the final sequence is unique and maintains a logical order.
        # This step is important if multiple triggers add the same persona or if
        # the insertion logic creates duplicates.
        seen = set()
        unique_sequence = []
        for persona in final_sequence:
            if persona not in seen:
                unique_sequence.append(persona)
                seen.add(persona)
        
        return unique_sequence