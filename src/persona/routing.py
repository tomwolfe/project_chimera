# src/persona/routing.py
"""
Dynamic persona routing system that selects appropriate personas
based on prompt analysis and intermediate results.
"""

from typing import List, Dict, Set, Optional, Any
import re
import json
from pathlib import Path
import logging

# Import necessary models and libraries
from src.models import PersonaConfig
# Assuming SentenceTransformer and util are available if sentence-transformers is installed
try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available. Semantic routing will be disabled.")

logger = logging.getLogger(__name__)

class PersonaRouter:
    """Determines the optimal sequence of personas for a given prompt."""
    
    def __init__(self, all_personas: Dict[str, PersonaConfig]):
        self.all_personas = all_personas
        
        # Domain keywords for fallback keyword matching
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

        # --- NEW ATTRIBUTES FOR SEMANTIC ROUTING ---
        self._embedding_model = None
        self._domain_embeddings = {}
        self._domain_examples = {
            "Software Engineering": ["refactor code", "design a scalable API", "fix a Python bug", "implement a feature", "database schema", "system architecture", "clean code", "technical debt"],
            "Business Strategy": ["market analysis", "business plan", "ROI", "marketing strategy", "startup funding", "competitive advantage", "financial forecast", "operations management"],
            "Scientific Research": ["quantum physics", "genetic engineering", "climate modeling", "experimental design", "hypothesis testing", "data analysis", "biological process", "chemical reaction"],
            "Creative Writing": ["novel plot", "character development", "story arc", "poetry analysis", "scriptwriting", "world-building", "narrative structure", "dialogue writing"],
            "Architecture (Physical)": ["design a house", "skyscraper construction", "urban planning", "building materials", "civil engineering", "interior design", "structural integrity", "construction project"] # Added for disambiguation
        }
        self._load_domain_embeddings()
        # --- END NEW ATTRIBUTES ---
    
    def _load_domain_embeddings(self):
        """Loads embeddings for domain examples."""
        if not _SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence-transformers not available. Skipping domain embedding loading.")
            return

        try:
            if not self._embedding_model:
                # Lazy load the model
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            for domain, examples in self._domain_examples.items():
                self._domain_embeddings[domain] = self._embedding_model.encode(examples, convert_to_tensor=True)
            logger.info("Domain embeddings loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading domain embeddings: {e}")
            self._embedding_model = None # Ensure model is None if loading fails

    def _analyze_prompt_domain(self, prompt: str) -> Set[str]:
        """Analyze prompt to determine relevant domains using semantic understanding."""
        prompt_lower = prompt.lower()
        
        # --- Enhanced Self-Analysis Detection ---
        self_analysis_keywords = [
            "chimera", "your code", "self-analysis", "codebase", "refactor this code", 
            "improve your logic", "analyze your performance", "optimize this process", 
            "self-improvement", "system analysis", "critique your own output", "your reasoning"
        ]
        
        is_self_analysis_prompt = any(keyword in prompt_lower for keyword in self_analysis_keywords)
        
        if is_self_analysis_prompt:
            logger.info("Detected self-analysis prompt. Prioritizing 'Software Engineering' domain.")
            return {"Software Engineering", "Self-Analysis"} # Special handling for self-analysis
        # --- End Detection ---

        # Fallback to keyword matching if sentence-transformers is not available
        if not self._embedding_model:
            logger.warning("Using keyword matching for domain analysis as sentence-transformers is not loaded.")
            return self._analyze_prompt_domain_keyword_fallback(prompt)

        # Semantic analysis using sentence-transformers
        try:
            prompt_embedding = self._embedding_model.encode([prompt], convert_to_tensor=True)
            domain_scores = {}
            
            for domain, embeddings in self._domain_embeddings.items():
                cosine_scores = util.pytorch_cos_sim(prompt_embedding, embeddings)[0]
                # Use the maximum similarity score for this domain
                domain_scores[domain] = max(cosine_scores).item()
            
            # Apply a threshold and filter out ambiguous cases (e.g., "building architect")
            threshold = 0.55 # Tunable parameter
            relevant_domains = set()
            
            for domain, score in domain_scores.items():
                if score > threshold:
                    # Specific disambiguation for "Architecture"
                    if domain == "Architecture (Physical)":
                        # Check if prompt contains software-related terms that might confuse it
                        if not any(sw_keyword in prompt_lower for sw_keyword in ["software", "code", "api", "system", "application", "framework"]):
                            relevant_domains.add(domain)
                    else:
                        relevant_domains.add(domain)
            
            # If no domains meet the threshold, use the highest scoring domain as a fallback
            if not relevant_domains and domain_scores:
                top_domain = max(domain_scores, key=domain_scores.get)
                relevant_domains = {top_domain}
                logger.debug(f"No domains above threshold, using fallback: {top_domain} ({domain_scores[top_domain]:.2f})")
            
            # Ensure "Software Engineering" is included if related terms are present, even if score is low
            if any(sw_keyword in prompt_lower for sw_keyword in ["software", "code", "api", "system", "application", "framework", "refactor", "bug", "test"]):
                relevant_domains.add("Software Engineering")

            return relevant_domains if relevant_domains else {"General"}
            
        except Exception as e:
            logger.error(f"Error during semantic domain analysis: {e}. Falling back to keyword matching.")
            # Fallback to keyword matching if semantic analysis fails
            return self._analyze_prompt_domain_keyword_fallback(prompt)

    def _analyze_prompt_domain_keyword_fallback(self, prompt: str) -> Set[str]:
        """Fallback keyword analysis for domain detection."""
        prompt_lower = prompt.lower()
        domains = set()
        for domain, config in self.domain_keywords.items():
            # Apply negative keyword filtering
            has_negative_match = any(keyword in prompt_lower for keyword in config.get("negative", []))
            if has_negative_match:
                continue
            # Check for positive keywords
            has_positive_match = any(keyword in prompt_lower for keyword in config.get("positive", []))
            if has_positive_match:
                domains.add(domain)
        return domains if domains else {"General"}
    
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
        """Determine optimal persona sequence, prioritizing self-analysis and dynamic adjustments."""
        prompt_lower = prompt.lower()
        
        # --- Enhanced Self-Analysis Detection ---
        self_analysis_keywords = [
            "chimera", "your code", "self-analysis", "codebase", "refactor this code", 
            "improve your logic", "analyze your performance", "optimize this process", 
            "self-improvement", "system analysis", "critique your own output", "your reasoning"
        ]
        
        is_self_analysis_prompt = any(keyword in prompt_lower for keyword in self_analysis_keywords)
        
        if is_self_analysis_prompt:
            logger.info("Detected self-analysis prompt. Using specialized persona sequence.")
            # Specialized sequence for self-improvement tasks, prioritizing code analysis and critique
            return [
                "Code_Architect",         # To analyze structure and design
                "Skeptical_Generator",    # To find flaws in current logic/code
                "Constructive_Critic",    # To suggest specific improvements
                "Test_Engineer",          # To ensure robustness of proposed changes
                "Impartial_Arbitrator",   # To synthesize findings into actionable steps
                "Devils_Advocate"         # To challenge the proposed self-improvements
            ]
        # --- End Enhanced Self-Analysis Detection ---
        
        # --- Domain-specific routing (original logic) ---
        domains = self._analyze_prompt_domain(prompt)
        base_sequence = self._get_domain_specific_personas(domains)
        
        core_order = ["Visionary_Generator", "Skeptical_Generator"]
        domain_experts = [p for p in base_sequence
                         if p not in core_order and p != "Impartial_Arbitrator"]
        
        final_sequence = core_order + domain_experts
        if "Impartial_Arbitrator" in base_sequence: # Ensure Arbitrator is included if it was in the base set
            final_sequence.append("Impartial_Arbitrator")
        
        # --- Dynamic adjustment based on intermediate results ---
        if intermediate_results and len(intermediate_results) > 2:
            # Analyze recent outputs for trigger keywords to add/reorder personas
            # Example: If a critique mentions "need more technical depth"
            recent_output_text = ""
            # Find the last persona output for analysis
            for step_name, result in reversed(list(intermediate_results.items())):
                if step_name.endswith("_Output") and isinstance(result, str):
                    recent_output_text = result.lower()
                    break
            
            if recent_output_text:
                # Check for phrases that suggest adding specific personas
                if "need more technical depth" in recent_output_text or "architectural concerns" in recent_output_text:
                    if "Code_Architect" not in final_sequence: final_sequence.append("Code_Architect")
                    if "Security_Auditor" not in final_sequence: final_sequence.append("Security_Auditor")
                if "business viability" in recent_output_text or "market impact" in recent_output_text:
                    if "Business_Strategist" not in final_sequence: final_sequence.append("Business_Strategist")
                if "testing concerns" in recent_output_text or "bug found" in recent_output_text:
                    if "Test_Engineer" not in final_sequence: final_sequence.append("Test_Engineer")
                if "deployment issues" in recent_output_text or "operational challenges" in recent_output_text:
                    if "DevOps_Engineer" not in final_sequence: final_sequence.append("DevOps_Engineer")
            
        # --- Minimal 80/20 Refinement for Framework Selection (as before) ---
        # Example: Prevent "building architect" misclassification
        if "Code_Architect" in final_sequence:
            if ("building architect" in prompt_lower or "construction architect" in prompt_lower) and \
               not any(sw_keyword in prompt_lower for sw_keyword in ["software", "code", "api", "system", "framework"]):
                logger.warning("Misclassification detected: 'building architect' prompt likely triggered Code_Architect. Removing it.")
                final_sequence.remove("Code_Architect")
                if "Generalist_Assistant" not in final_sequence:
                    final_sequence.append("Generalist_Assistant")
        
        # Ensure uniqueness while preserving order
        seen = set()
        return [p for p in final_sequence if not (p in seen or seen.add(p))]