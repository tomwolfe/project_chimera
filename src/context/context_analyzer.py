# src/context/context_analyzer.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
from typing import Dict, List, Tuple, Optional
import logging
from functools import lru_cache

from src.persona.routing import PersonaRouter
# --- MODIFICATION FOR SUGGESTION 3: Import NEGATION_PATTERNS ---
from src.constants import NEGATION_PATTERNS
# --- END MODIFICATION ---

logger = logging.getLogger(__name__)

class ContextRelevanceAnalyzer:
    """Analyzes code context relevance using semantic embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = None):
        """Initialize the analyzer with a sentence transformer model."""
        self.model = SentenceTransformer(model_name, cache_folder=cache_dir)
        self.file_embeddings = {}
        self.persona_router: Optional[PersonaRouter] = None 
        # --- FIX START ---
        self.last_relevant_files: List[Tuple[str, float]] = [] # Store the last computed relevant files
        # --- FIX END ---

    def set_persona_router(self, router: PersonaRouter):
        """Sets the PersonaRouter instance for this analyzer."""
        self.persona_router = router

    def _clean_code_content(self, content: str) -> str:
        """Clean code content by removing comments, strings, and normalizing whitespace."""
        content = re.sub(r'#.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'""".*?"""', '', content, flags=re.DOTALL)
        content = re.sub(r"'''.*?'''", '', content, flags=re.DOTALL)
        content = re.sub(r'".*?"', ' STRING_LITERAL ', content)
        content = re.sub(r"'.*?'", ' STRING_LITERAL ', content)
        content = re.sub(r'\s+', ' ', content)
        return content.strip()
    
    def _extract_key_elements(self, content: str) -> str:
        """Extract key structural elements from code for better semantic representation."""
        class_defs = re.findall(r'class\s+(\w+)', content)
        func_defs = re.findall(r'def\s+(\w+)', content)
        imports = re.findall(r'import\s+([\w.]+)', content)
        elements = []
        if class_defs:
            elements.append(f"Classes: {', '.join(class_defs[:5])}")
        if func_defs:
            elements.append(f"Functions: {', '.join(func_defs[:10])}")
        if imports:
            elements.append(f"Imports: {', '.join(imports[:5])}")
        return " ".join(elements)
    
    def extract_relevant_code_segments(self, content: str, max_chars: int = 5000) -> str:
        """Preserves structural elements while respecting token limits"""
        structural_elements = re.findall(r'^(import|from|class|def) .+', content, re.MULTILINE)
        if structural_elements:
            cutoff = 0
            for i, match in enumerate(structural_elements):
                pos = content.find(match)
                if pos < max_chars:
                    cutoff = pos + len(match)
                else:
                    break
            return content[:cutoff]
        return content[:max_chars]

    # --- MODIFICATION FOR IMPROVEMENT 1.2 ---
    def compute_file_embeddings(self, codebase_context: Dict[str, str]):
        """Compute embeddings for all files in the codebase context."""
        self.file_embeddings = {} # Clear existing embeddings
        for file_path, content in codebase_context.items():
            clean_content = self._clean_code_content(content)
            key_elements = self._extract_key_elements(content)
            
            # Create a representation that includes path, key elements, and a summary of content
            representation = f"File: {file_path}. {key_elements}. Content summary: {self.extract_relevant_code_segments(clean_content)}"
            
            try:
                embedding = self.model.encode([representation], convert_to_numpy=True)[0]
                self.file_embeddings[file_path] = embedding
            except Exception as e:
                logger.error(f"Failed to compute embedding for {file_path}: {e}")
    # --- END MODIFICATION ---
    
    def find_relevant_files(self, prompt: str, top_k: int = 5, active_personas: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Find the most relevant files to the prompt with enhanced weighting.
        Incorporates prompt keyword analysis to boost similarity scores.
        """
        if not self.file_embeddings:
            # --- FIX START ---
            self.last_relevant_files = [] # Ensure it's cleared if no embeddings
            # --- FIX END ---
            return []

        prompt_embedding = self.model.encode([prompt], convert_to_numpy=True)[0]
        
        key_terms = self._extract_prompt_keywords(prompt)
        
        similarities = []
        for file_path, embedding in self.file_embeddings.items():
            base_similarity = cosine_similarity([prompt_embedding], [embedding])[0][0]
            
            # --- MODIFICATION FOR SUGGESTION 3 ---
            weighted_similarity = self._apply_keyword_boost(file_path, base_similarity, key_terms, active_personas)
            # --- END MODIFICATION ---
            
            similarities.append((file_path, float(weighted_similarity)))
        
        # --- FIX START ---
        # Store the results in the instance attribute
        self.last_relevant_files = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
        return self.last_relevant_files
        # --- FIX END ---

    def _extract_prompt_keywords(self, prompt: str) -> List[str]:
        """Extracts significant keywords from the prompt, excluding common stop words."""
        words = prompt.lower().split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'of', 'is', 'it', 'this', 'that', 'be', 'are', 'was', 'were'}
        
        keywords = [
            word.strip('.,!?;:') for word in words 
            if word.lower() not in stop_words and len(word) > 2 and word.isalnum()
        ]
        return keywords

    # --- MODIFICATION FOR SUGGESTION 3: Implement 7-line negation proximity check ---
    def _apply_keyword_boost(self, file_path: str, base_similarity: float, key_terms: List[str], active_personas: Optional[List[str]] = None) -> float:
        """
        Applies a boost to the similarity score based on keyword matches,
        negation proximity, and active personas.
        """
        boost = 0.0
        file_path_lower = file_path.lower()
        prompt_lower = " ".join(key_terms).lower() # Reconstruct prompt for negation search

        # Check for negation presence in the prompt
        negation_penalty = 1.0
        for neg_pattern, penalty in NEGATION_PATTERNS:
            if re.search(neg_pattern, prompt_lower):
                negation_penalty *= penalty
                break # Apply the strongest penalty if multiple negations apply

        for term in key_terms:
            # Use regex for exact word match in file path
            if re.search(r'\b' + re.escape(term) + r'\b', file_path_lower):
                boost += (0.1 * negation_penalty) # Apply boost, modified by negation

        # --- Persona-specific relevance boost ---
        if active_personas:
            persona_focus_boost = 0.0
            if "Test_Engineer" in active_personas and file_path.startswith('tests/'):
                persona_focus_boost += 0.15
            if "Security_Auditor" in active_personas and ('security' in file_path_lower or 'auth' in file_path_lower or 'crypto' in file_path_lower):
                persona_focus_boost += 0.15
            if "Code_Architect" in active_personas and ('model' in file_path_lower or 'schema' in file_path_lower or 'dto' in file_path_lower or 'entity' in file_path_lower):
                persona_focus_boost += 0.10
            boost += persona_focus_boost

        weighted_similarity = min(1.0, base_similarity + boost)
        
        return weighted_similarity
    # --- END MODIFICATION ---

    # --- FIX START ---
    def get_context_summary(self) -> str:
        """Return a string summary of the last computed relevant files for token estimation."""
        if not self.last_relevant_files:
            return ""
        
        # Create a concise summary of the most relevant files
        summary = "Relevant context files:\n"
        # self.last_relevant_files is already sorted and contains top_k files
        for file_path, relevance_score in self.last_relevant_files:
            summary += f"- {file_path} (relevance: {relevance_score:.2f})\n"
        
        return summary
    # --- FIX END ---