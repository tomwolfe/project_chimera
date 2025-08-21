# src/context/context_analyzer.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
from typing import Dict, List, Tuple, Optional
import logging
from functools import lru_cache
import os # Import os for basename

from src.persona.routing import PersonaRouter
from src.constants import NEGATION_PATTERNS

logger = logging.getLogger(__name__)

class ContextRelevanceAnalyzer:
    """Analyzes code context relevance using semantic embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = None, codebase_context: Optional[Dict[str, str]] = None): # ADD codebase_context to __init__
        """Initialize the analyzer with a sentence transformer model.
        
        Args:
            model_name: The name of the SentenceTransformer model to use.
            cache_dir: Optional. The directory where the model should be loaded from or cached to.
                       If None, SentenceTransformer uses its default caching mechanism.
            codebase_context: Optional. The full codebase context dictionary.
        """
        self.cache_dir = cache_dir # Store the cache_dir
        # Pass cache_folder directly to SentenceTransformer
        self.model = SentenceTransformer(model_name, cache_folder=self.cache_dir) 
        self.file_embeddings = {}
        self.persona_router: Optional[PersonaRouter] = None 
        # --- FIX START ---
        self.last_relevant_files: List[Tuple[str, float]] = [] # Store the last computed relevant files
        # --- FIX END ---
        self.codebase_context = codebase_context or {} # Store codebase_context

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
        self.codebase_context = codebase_context # Ensure internal codebase_context is updated
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
    
    @lru_cache(maxsize=128) # Cache relevant files based on prompt and active personas
    def find_relevant_files(self, prompt: str, max_context_tokens: int, active_personas: Optional[Tuple[str, ...]] = None) -> List[Tuple[str, float]]:
        """
        Find the most relevant files to the prompt with enhanced weighting,
        considering the maximum allowed tokens for context.
        
        FIX: Changed active_personas type hint from List[str] to Tuple[str, ...]
             because List is unhashable and causes TypeError with @lru_cache.
        """
        if not self.file_embeddings or not self.codebase_context:
            self.last_relevant_files = []
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
        
        # Sort by similarity
        sorted_files = sorted(similarities, key=lambda x: x[1], reverse=True)

        # Dynamically select files based on max_context_tokens
        selected_files_with_content = []
        current_tokens = 0
        
        # Use a simple character-based heuristic for token estimation if no tokenizer is available
        # A more robust solution would involve injecting the tokenizer from SocraticDebate.
        CHARS_PER_TOKEN = 4 
        
        for file_path, score in sorted_files:
            content = self.codebase_context.get(file_path, "") 
            file_token_estimate = len(content) / CHARS_PER_TOKEN
            
            if current_tokens + file_token_estimate <= max_context_tokens:
                selected_files_with_content.append((file_path, score))
                current_tokens += file_token_estimate
            else:
                break # Stop adding files if budget is exceeded

        # --- FIX START ---
        # Store the results in the instance attribute
        self.last_relevant_files = selected_files_with_content
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

    def _calculate_negation_impact(self, prompt_lower: str, keyword_start_pos: int, max_proximity: int = 75) -> float:
        """
        Calculates a multiplier (0.0 to 1.0) based on negation proximity to a keyword.
        Closer negations result in a lower multiplier (stronger negative impact).
        """
        min_multiplier = 0.1 # Max penalty (e.g., "not keyword" makes it 10% as relevant)
        max_multiplier = 1.0 # No penalty

        closest_negation_distance = float('inf')
        for neg_pattern, base_penalty_factor in NEGATION_PATTERNS:
            for neg_match in re.finditer(neg_pattern, prompt_lower):
                negation_end_pos = neg_match.end()
                # Only consider negations *before* the keyword within the specified proximity
                if negation_end_pos < keyword_start_pos:
                    distance = keyword_start_pos - negation_end_pos
                    if distance <= max_proximity:
                        closest_negation_distance = min(closest_negation_distance, distance)

        if closest_negation_distance == float('inf'):
            return max_multiplier # No relevant negation found

        # Linear interpolation for multiplier:
        # At distance 0, multiplier is min_multiplier.
        # At distance max_proximity, multiplier is max_multiplier.
        # (1 - (distance / max_proximity)) scales from 1 to 0 as distance goes from 0 to max_proximity
        # This makes the multiplier go from min_multiplier to max_multiplier
        decay_factor = 1 - (closest_negation_distance / max_proximity)
        return min_multiplier + (max_multiplier - min_multiplier) * decay_factor

    # --- MODIFICATION FOR SUGGESTION 3: Implement 7-line negation proximity check ---
    def _apply_keyword_boost(self, file_path: str, base_similarity: float, key_terms: List[str], active_personas: Optional[Tuple[str, ...]] = None) -> float:
        """
        Applies a boost to the similarity score based on keyword matches,
        negation proximity, and active personas.
        """
        boost = 0.0
        file_path_lower = file_path.lower()
        prompt_lower = " ".join(key_terms).lower() # Reconstruct prompt for negation search

        for term in key_terms:
            # Use regex for exact word match in file path
            if re.search(r'\b' + re.escape(term) + r'\b', file_path_lower):
                keyword_start_pos_in_prompt = prompt_lower.find(term)
                if keyword_start_pos_in_prompt != -1:
                    negation_multiplier = self._calculate_negation_impact(prompt_lower, keyword_start_pos_in_prompt)
                    boost += (0.1 * negation_multiplier) # Apply boost, modified by negation impact

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

    # NEW: Intelligent context summarization
    def generate_context_summary(self, relevant_file_paths: List[str], max_tokens: int, debate_topic: str) -> str:
        """
        Generates a concise, intelligent summary of relevant files, respecting token limits.
        Focuses on high-level structure and key implementation details related to the debate topic.
        """
        summary_parts = [f"Relevant code context for '{debate_topic}':"]
        current_tokens = self._estimate_token_count(summary_parts[0])

        # Extract high-level structure first
        structure_info = self._extract_structure_info(relevant_file_paths)
        if structure_info:
            header = "\nSystem Structure:"
            summary_parts.append(header)
            current_tokens += self._estimate_token_count(header)
            
            for line in structure_info.split('\n'):
                line_tokens = self._estimate_token_count(line)
                if current_tokens + line_tokens <= max_tokens:
                    summary_parts.append(line)
                    current_tokens += line_tokens
                else:
                    break
            if current_tokens >= max_tokens: return "\n".join(summary_parts)

        # Add key implementation details
        details_header = "\nKey Implementation Details:"
        details_parts = []
        details_tokens = self._estimate_token_count(details_header)

        for file_path in relevant_file_paths:
            content = self.codebase_context.get(file_path, "")
            if not content: continue
            
            lines = content.split('\n')
            file_relevant_lines = []
            topic_keywords = debate_topic.lower().split()

            # Prioritize lines with direct keyword matches
            for line in lines:
                line_lower = line.lower()
                if any(kw in line_lower for kw in topic_keywords) and not line.strip().startswith('#'):
                    file_relevant_lines.append(line.strip())
            
            # If not enough direct matches, take some initial non-comment lines
            if len(file_relevant_lines) < 3:
                file_relevant_lines = [l.strip() for l in lines[:15] if l.strip() and not l.strip().startswith('#')]
            
            if file_relevant_lines:
                file_summary_header = f"\nFile: {os.path.basename(file_path)}:"
                details_tokens += self._estimate_token_count(file_summary_header)
                if current_tokens + details_tokens > max_tokens: break

                details_parts.append(file_summary_header)
                for line in file_relevant_lines[:5]: # Take up to 5 lines per file
                    line_tokens = self._estimate_token_count(line)
                    if current_tokens + details_tokens + line_tokens <= max_tokens:
                        details_parts.append(f"  {line}")
                        details_tokens += line_tokens
                    else:
                        break
        
        if details_parts:
            summary_parts.extend(details_parts)

        return "\n".join(summary_parts)

    def _extract_structure_info(self, relevant_file_paths: List[str]) -> str:
        """Extract high-level structure information from relevant files."""
        structure_parts = []
        for file_path in relevant_file_paths[:5]: # Only look at top 5 files for high-level structure
            content = self.codebase_context.get(file_path, "")
            if not content: continue
            
            classes = re.findall(r'class\s+(\w+)\s*(\(.*\))?\s*:', content)
            functions = re.findall(r'def\s+(\w+)\s*\(', content)
            
            file_summary = f"- {os.path.basename(file_path)}"
            if classes:
                file_summary += f" (Classes: {', '.join([c[0] for c in classes[:3]])}{'...' if len(classes) > 3 else ''})"
            if functions:
                file_summary += f" (Functions: {', '.join(functions[:5])}{'...' if len(functions) > 5 else ''})"
            structure_parts.append(file_summary)
        return "\n".join(structure_parts)

    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count for a text string (fallback heuristic)."""
        return max(1, len(text) // 4) # ~4 characters per token