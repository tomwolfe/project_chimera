# src/context/context_analyzer.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
from typing import Dict, List, Tuple

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
    
    # --- NEW FUNCTION INSERTION START ---
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
    # --- NEW FUNCTION INSERTION END ---

    def compute_file_embeddings(self, codebase_context: Dict[str, str]):
        """Compute embeddings for all files in the codebase context."""
        for file_path, content in codebase_context.items():
            # Create a meaningful representation of the file
            clean_content = self._clean_code_content(content)
            key_elements = self._extract_key_elements(content)
            
            # --- MODIFIED LINE START ---
            # Replace: representation = f"File: {file_path}. {key_elements}. Content summary: {clean_content[:500]}"
            representation = f"File: {file_path}. {key_elements}. Content summary: {self.extract_relevant_code_segments(clean_content)}"
            # --- MODIFIED LINE END ---
            
            # Generate embedding
            embedding = self.model.encode([representation], convert_to_numpy=True)[0]
            self.file_embeddings[file_path] = embedding
    
    def find_relevant_files(self, prompt: str, top_k: int = 5) -> List[Tuple[str, float]]:
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
            weighted_similarity = self._apply_keyword_boost(file_path, base_similarity, key_terms)
            
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

    def _apply_keyword_boost(self, file_path: str, base_similarity: float, key_terms: List[str]) -> float:
        """
        Applies a boost to the similarity score based on keyword matches in the file path.
        This function can be extended for more sophisticated relevance scoring.
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
        
        # Example: If prompt mentions 'test' and file path starts with 'test_'
        if 'test' in key_terms and file_path.startswith('tests/'):
            boost += 0.2

        # Combine base similarity with boost, capping at 1.0
        weighted_similarity = min(1.0, base_similarity + boost)
        
        return weighted_similarity