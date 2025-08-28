# src/self_improvement/content_validator.py
import re
import logging
import json
from typing import List, Tuple, Dict, Any, Union, Optional

logger = logging.getLogger(__name__)

class ContentAlignmentValidator:
    """
    Validates if a persona's output aligns with the original prompt's focus areas.
    Designed to prevent content drift during the debate process.
    """
    def __init__(self, original_prompt: str, debate_domain: str, focus_areas: Optional[List[str]] = None):
        self.original_prompt = original_prompt.lower()
        self.debate_domain = debate_domain.lower()
        
        # Define default focus areas if not explicitly provided, based on self-improvement context
        if focus_areas is None:
            if self.debate_domain == "self-improvement":
                self.focus_areas = [
                    "reasoning quality", "robustness", "efficiency", "maintainability",
                    "code changes", "process adjustments", "project chimera codebase",
                    "pep8", "code smells", "security vulnerabilities", "token usage",
                    "schema validation", "conflict resolution", "test coverage"
                ]
            elif self.debate_domain == "software engineering":
                self.focus_areas = [
                    "code", "implement", "refactor", "bug fix", "architecture", "security", "testing", "devops",
                    "api", "database", "function", "class", "module", "performance", "scalability"
                ]
            else:
                # For other domains, extract keywords from the original prompt itself
                # This is a basic heuristic and can be refined.
                self.focus_areas = self._extract_keywords_from_prompt(self.original_prompt)
        else:
            self.focus_areas = [area.lower() for area in focus_areas]
        
        logger.info(f"ContentAlignmentValidator initialized for domain '{self.debate_domain}' with focus areas: {self.focus_areas}")

    def _extract_keywords_from_prompt(self, prompt: str, num_keywords: int = 5) -> List[str]:
        """Extracts significant keywords from the prompt to form dynamic focus areas."""
        words = re.findall(r'\b\w+\b', prompt.lower())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'of', 'is', 'it', 'this', 'that', 'be', 'are', 'was', 'were', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how', 'do', 'does', 'did', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'can', 'could', 'will', 'would', 'shall', 'should', 'may', 'might', 'must'}
        
        # Filter out stop words and short words, then count frequency
        word_counts = {}
        for word in words:
            if word not in stop_words and len(word) > 2:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Get top N most frequent words as keywords
        sorted_keywords = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
        return [kw[0] for kw in sorted_keywords[:num_keywords]]

    def validate(self, persona_name: str, persona_output: Union[str, Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Checks if the persona's output aligns with the defined focus areas.
        
        Args:
            persona_name: The name of the persona generating the output.
            persona_output: The output from the persona (can be string or dict).
            
        Returns:
            Tuple[bool, str]: (is_aligned, validation_message)
        """
        output_text = ""
        if isinstance(persona_output, dict):
            # Extract relevant text from structured output
            if persona_name == "Constructive_Critic" and "CRITIQUE_SUMMARY" in persona_output:
                output_text = persona_output["CRITIQUE_SUMMARY"]
            elif persona_name == "Devils_Advocate" and "summary" in persona_output:
                output_text = persona_output["summary"]
            elif persona_name == "Self_Improvement_Analyst" and "ANALYSIS_SUMMARY" in persona_output:
                output_text = persona_output["ANALYSIS_SUMMARY"]
            elif "general_output" in persona_output:
                output_text = persona_output["general_output"]
            else:
                # Fallback: convert dict to string for general keyword search
                output_text = json.dumps(persona_output)
        else:
            output_text = str(persona_output)
        
        output_text_lower = output_text.lower()

        if not self.focus_areas:
            logger.debug(f"No specific focus areas defined for domain '{self.debate_domain}'. Content validation skipped for {persona_name}.")
            return True, "No specific focus areas defined for this domain."

        # At least one focus area must be present in the output
        found_focus_area = False
        for area in self.focus_areas:
            if area in output_text_lower:
                found_focus_area = True
                break
        
        if not found_focus_area:
            return False, f"Output from {persona_name} does not sufficiently address the core focus areas: {', '.join(self.focus_areas[:3])}..."

        # Additionally, check for strong negative indicators (e.g., discussing unrelated topics too much)
        # These are examples of topics from other example prompts.
        negative_keywords = ["mars city", "ethical ai framework", "climate change solution", "fastapi endpoint"] 
        for neg_kw in negative_keywords:
            if neg_kw in output_text_lower and "project chimera" not in output_text_lower:
                return False, f"Output from {persona_name} appears to be discussing an unrelated topic: '{neg_kw}'."

        return True, "Content aligned with focus areas."