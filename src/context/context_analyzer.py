# src/context/context_analyzer.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
from typing import Dict, List, Tuple, Optional # Added Optional
import logging # Ensure logging is imported
from functools import lru_cache # Import lru_cache

# NEW: Import PersonaRouter to access its methods
from src.persona.routing import PersonaRouter # CORRECTED PATH

logger = logging.getLogger(__name__)

class ContextRelevanceAnalyzer:
    """Analyzes code context relevance using semantic embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = None):
        """Initialize the analyzer with a sentence transformer model."""
        self.model = SentenceTransformer(model_name, cache_folder=cache_dir)
        self.file_embeddings = {}
        # NOTE: To make _apply_keyword_boost more effective, we would ideally store
        # file contents or extracted key elements here. For this revision, we'll
        # focus on analyzing the file path and prompt keywords directly for simplicity.
        # If file contents were stored, they would be loaded here or passed to methods.
        
        # Placeholder for PersonaRouter instance. This should be injected via a setter
        # or constructor argument in a real application to manage dependencies properly.
        self.persona_router: Optional[PersonaRouter] = None 

    def set_persona_router(self, router: PersonaRouter):
        """Sets the PersonaRouter instance for this analyzer."""
        self.persona_router = router

    def _clean_code_content(self, content: str) -> str:
        """Clean code content by removing comments, strings, and normalizing whitespace."""
        # Remove single-line comments
        content = re.sub(r'#.*$', '', content, flags=re.MULTILINE)
        # Remove multi-line comments/docstrings
        content = re.sub(r'""".*?"""', '', content, flags=re.DOTALL)
        content = re.sub(r"'''.*?'''", '', content, flags=re.DOTALL)
        # Replace strings with placeholders
        content = re.sub(r'".*?"', ' STRING_LITERAL ', content)
        content = re.sub(r"'.*?'", ' STRING_LITERAL ', content)
        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content)
        return content.strip()
    
    def _extract_key_elements(self, content: str) -> str:
        """Extract key structural elements from code for better semantic representation."""
        # Extract class and function definitions
        class_defs = re.findall(r'class\s+(\w+)', content)
        func_defs = re.findall(r'def\s+(\w+)', content)
        # Extract import statements
        imports = re.findall(r'import\s+([\w.]+)', content)
        # Return key elements as a descriptive string
        elements = []
        if class_defs:
            elements.append(f"Contains classes: {', '.join(class_defs[:5])}")
        if func_defs:
            elements.append(f"Contains functions: {', '.join(func_defs[:10])}")
        if imports:
            elements.append(f"Imports: {', '.join(imports[:5])}")
        return " ".join(elements)
    
    def extract_relevant_code_segments(self, content: str, max_chars: int = 5000) -> str:
        """Preserves structural elements while respecting token limits"""
        # Keep imports and class/function definitions at top
        structural_elements = re.findall(r'^(import|from|class|def) .+', content, re.MULTILINE)
        if structural_elements:
            # Find position of last structural element within limit
            cutoff = 0
            for i, match in enumerate(structural_elements): # Using the exact loop from prompt
                pos = content.find(match)
                if pos < max_chars:
                    cutoff = pos + len(match)
                else:
                    break
            return content[:cutoff]
        return content[:max_chars]

    def compute_file_embeddings(self, codebase_context: Dict[str, str]):
        """Compute embeddings for all files in the codebase context."""
        for file_path, content in codebase_context.items():
            # Create a meaningful representation of the file
            clean_content = self._clean_code_content(content)
            key_elements = self._extract_key_elements(content)
            
            representation = f"File: {file_path}. {key_elements}. Content summary: {self.extract_relevant_code_segments(clean_content)}"
            
            # Generate embedding
            embedding = self.model.encode([representation], convert_to_numpy=True)[0]
            self.file_embeddings[file_path] = embedding
    
    def find_relevant_files(self, prompt: str, top_k: int = 5, active_personas: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Find the most relevant files to the prompt with enhanced weighting.
        Incorporates prompt keyword analysis to boost similarity scores.
        """
        if not self.file_embeddings:
            return []

        prompt_embedding = self.model.encode([prompt], convert_to_numpy=True)[0]
        
        # 1. Extract key terms from the prompt for keyword analysis
        key_terms = self._extract_prompt_keywords(prompt)
        
        similarities = []
        for file_path, embedding in self.file_embeddings.items():
            # Calculate base similarity using embeddings
            base_similarity = cosine_similarity([prompt_embedding], [embedding])[0][0]
            
            # 2. Apply keyword-based relevance boost
            # Pass active_personas to the boost method
            weighted_similarity = self._apply_keyword_boost(file_path, base_similarity, key_terms, active_personas)
            
            similarities.append((file_path, float(weighted_similarity)))
        
        # Sort by the final weighted similarity score (descending)
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

    def _extract_prompt_keywords(self, prompt: str) -> List[str]:
        """Extracts significant keywords from the prompt, excluding common stop words."""
        words = prompt.lower().split()
        # Simple stop word list for common English words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'of', 'is', 'it', 'this', 'that', 'be', 'are', 'was', 'were'}
        
        # Extract words that are likely keywords (alphanumeric, longer than 2 chars, not stop words)
        keywords = [
            word.strip('.,!?;:') for word in words 
            if word.lower() not in stop_words and len(word) > 2 and word.isalnum()
        ]
        return keywords

    # --- MODIFIED METHOD FOR REASONING QUALITY (#1 PRIORITY) ---
    def _apply_keyword_boost(self, file_path: str, base_similarity: float, key_terms: List[str], active_personas: Optional[List[str]] = None) -> float:
        """
        Applies a boost to the similarity score based on keyword matches in the file path,
        semantic relevance of file content/key elements, and active personas.
        """
        boost = 0.0
        file_path_lower = file_path.lower()
        
        # Boost based on keywords appearing in the file path itself
        for term in key_terms:
            if term in file_path_lower:
                boost += 0.1 # Small boost for path matches

        # Example: If prompt mentions 'API' and file path contains 'controller' or 'service'
        if 'api' in key_terms and ('controller' in file_path_lower or 'service' in file_path_lower or 'route' in file_path_lower):
            boost += 0.15
        
        # Boost for test files if prompt indicates testing/debugging needs
        if any(kw in key_terms for kw in ["test", "debug", "bug", "quality", "coverage", "refactor", "code"]):
            if file_path.startswith('tests/'):
                boost += 0.2 # Apply the specified 0.2 boost for test files
        
        # --- NEW: Semantic relevance boost based on file content/key elements ---
        # This requires access to file content or pre-computed embeddings of key elements.
        # For simplicity, we'll assume a method `_get_semantic_relevance` exists that
        # uses embeddings or keyword analysis of file content.
        # Example:
        # file_content_summary = self.get_file_summary(file_path) # Hypothetical method
        # semantic_relevance_score = self._calculate_semantic_relevance(key_terms, file_content_summary)
        # boost += semantic_relevance_score * 0.3 # Add a boost based on semantic match

        # --- NEW: Persona-specific relevance boost ---
        if active_personas:
            persona_focus_boost = 0.0
            if "Test_Engineer" in active_personas and file_path.startswith('tests/'):
                persona_focus_boost += 0.15
            if "Security_Auditor" in active_personas and ('security' in file_path_lower or 'auth' in file_path_lower or 'crypto' in file_path_lower):
                persona_focus_boost += 0.15
            if "Code_Architect" in active_personas and ('model' in file_path_lower or 'schema' in file_path_lower or 'dto' in file_path_lower or 'entity' in file_path_lower):
                persona_focus_boost += 0.10
            boost += persona_focus_boost

        # Combine base similarity with boost, capping at 1.0
        weighted_similarity = min(1.0, base_similarity + boost)
        
        return weighted_similarity
    # --- END MODIFIED METHOD ---

    # --- MODIFIED METHOD FOR #2 PRIORITY (ROBUSTNESS) ---
    # This method implements the context_ratio formula as requested in the prompt.
    def _get_adaptive_phase_ratios(self, prompt: str, context_present: bool) -> Dict[str, float]:
        """
        Dynamically adjust phase ratios using the context_ratio formula for Robustness.
        This ensures stable token allocation based on prompt complexity.
        """
        # Base ratio from settings or default
        base_ratio = 0.15 # Default base ratio, could be configurable
        
        # Calculate complexity score for the prompt
        complexity_score = self._calculate_complexity_score(prompt)
        
        # Apply the formula: context_ratio = max(0.1, min(0.3, base_ratio + complexity_score * 0.05))
        # This ensures context ratio is between 10% and 30% of available tokens.
        context_ratio = max(0.1, min(0.3, base_ratio + complexity_score * 0.05))
        
        # The remaining budget is split between debate and synthesis.
        # The prompt implies a split, but doesn't specify exact ratios for debate/synthesis.
        # A common split is 85% to debate, 15% to synthesis.
        remaining_budget_share = 1.0 - context_ratio
        debate_ratio = remaining_budget_share * 0.85
        synthesis_ratio = remaining_budget_share * 0.15
        
        logger.info(f"Calculated phase ratios: Context={context_ratio:.2%}, Debate={debate_ratio:.2%}, Synthesis={synthesis_ratio:.2%}")
        
        return {
            "context": context_ratio,
            "debate": debate_ratio,
            "synthesis": synthesis_ratio
        }
    # --- END MODIFIED METHOD ---

    # --- NEW HELPER FUNCTION FOR #2 PRIORITY (ROBUSTNESS) ---
    def _calculate_complexity_score(self, prompt: str) -> float:
        """
        Calculate a semantic complexity score for the prompt (0.0 to 1.0).
        This score influences the dynamic allocation of token budgets.
        """
        prompt_lower = prompt.lower()
        complexity = 0.0
        
        # Factor 1: Prompt length (normalized)
        # Longer prompts are generally more complex.
        length_factor = min(1.0, len(prompt) / 2000.0) # Normalize length to a 0-1 scale
        complexity += length_factor * 0.5 # Contribute up to 0.5 to complexity
        
        # Factor 2: Presence of technical keywords
        # Keywords related to code, analysis, or specific domains increase complexity.
        technical_keywords = [
            "code", "analyze", "refactor", "algorithm", "architecture", "system",
            "science", "research", "business", "market", "creative", "art",
            "security", "test", "deploy", "optimize", "debug"
        ]
        keyword_count = sum(1 for kw in technical_keywords if kw in prompt_lower)
        keyword_density = keyword_count / len(technical_keywords) if technical_keywords else 0
        complexity += keyword_density * 0.5 # Contribute up to 0.5 for keyword density
        
        # Ensure complexity score is within the [0.0, 1.0] range
        return max(0.0, min(1.0, complexity))
    # --- END NEW HELPER FUNCTION ---