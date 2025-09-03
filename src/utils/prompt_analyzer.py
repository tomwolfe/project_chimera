# File: src/utils/prompt_analyzer.py
# Modification: Add the definition for `optimize_reasoning_prompt`.

import re
import logging
from functools import lru_cache
from typing import Dict, Any, List, Optional, Tuple

# Assuming SELF_ANALYSIS_KEYWORDS, NEGATION_PATTERNS, and THRESHOLD are imported from src.constants
# These are global constants, so importing them directly is appropriate.
from src.constants import SELF_ANALYSIS_KEYWORDS, NEGATION_PATTERNS, THRESHOLD
# from src.tokenizers.base import Tokenizer # Assuming Tokenizer might be needed for optimization logic, but not strictly for this fix

logger = logging.getLogger(__name__)


class PromptAnalyzer:
    def __init__(self, domain_keywords: Dict[str, List[str]]):
        """
        Initializes the PromptAnalyzer with domain-specific keywords.

        Args:
            domain_keywords: A dictionary mapping domain names to lists of keywords.
        """
        self.domain_keywords = domain_keywords

    @lru_cache(maxsize=256)
    def analyze_complexity(self, prompt: str) -> Dict[str, Any]:
        """
        Analyzes prompt complexity, including word count, sentence count,
        and domain-specific keyword scores.

        Args:
            prompt: The user's input prompt.

        Returns:
            A dictionary containing complexity metrics and domain scores.
        """
        word_count = len(prompt.split())
        sentence_count = len(re.findall(r"[.!?]+", prompt))

        prompt_lower = prompt.lower()
        domain_scores = {
            domain: sum(1 for kw in keywords if kw in prompt_lower)
            for domain, keywords in self.domain_keywords.items()
        }

        # Determine primary domain based on highest score
        primary_domain = (
            max(domain_scores, key=domain_scores.get)
            if max(domain_scores.values()) > 0
            else None
        )

        # Simple complexity score calculation
        complexity_score = min(1.0, word_count / 500 + sentence_count / 20)

        # Add reasoning quality metrics
        reasoning_indicators = {
            "contains_80_20_language": "80/20" in prompt or "Pareto" in prompt.lower(),
            "explicit_focus_areas": any(area in prompt.lower() for area in
                ["reasoning quality", "robustness", "efficiency", "maintainability"]),
            "token_usage_warning": "token usage" in prompt.lower() or "cost" in prompt.lower(),
            "structured_output_request": "JSON" in prompt or "schema" in prompt.lower()
        }

        # Calculate reasoning quality score (0-1)
        reasoning_score = (
            0.25 * reasoning_indicators["contains_80_20_language"] +
            0.25 * reasoning_indicators["explicit_focus_areas"] +
            0.25 * reasoning_indicators["token_usage_warning"] +
            0.25 * reasoning_indicators["structured_output_request"]
        )

        return {
            "complexity_score": complexity_score,
            "primary_domain": primary_domain,
            "domain_scores": domain_scores,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "reasoning_quality_metrics": {
                "score": reasoning_score,
                "indicators": reasoning_indicators,
                "suggested_improvements": self._generate_reasoning_quality_suggestions(reasoning_indicators)
            }
        }

    def _generate_reasoning_quality_suggestions(self, indicators: Dict[str, bool]) -> List[str]:
        """Generates suggestions for improving reasoning quality based on indicators."""
        suggestions = []
        if not indicators["contains_80_20_language"]:
            suggestions.append("Add explicit 80/20 Pareto principle directive.")
        if not indicators["explicit_focus_areas"]:
            suggestions.append("Specify core focus areas (e.g., reasoning quality, robustness, efficiency, maintainability).")
        if not indicators["token_usage_warning"]:
            suggestions.append("Include directives for conciseness and token consciousness.")
        if not indicators["structured_output_request"]:
            suggestions.append("Request strict adherence to JSON schema for output.")
        return suggestions

    @lru_cache(maxsize=256)
    def is_self_analysis_prompt(
        self, prompt: str, threshold: float = THRESHOLD, negation_proximity: int = 100
    ) -> bool:
        """
        Checks if a prompt indicates a self-analysis task based on weighted keywords,
        positional boosting, and negation detection.

        Args:
            prompt: The user's input prompt.
            threshold: The score threshold to qualify as a self-analysis prompt.
            negation_proximity: The character proximity to check for negating words.

        Returns:
            True if the prompt is classified as self-analysis, False otherwise.
        """
        if not prompt:
            return False

        prompt_lower = prompt.lower()
        score = 0.0

        for keyword, weight in SELF_ANALYSIS_KEYWORDS.items():
            keyword_pos = prompt_lower.find(keyword)
            if keyword_pos != -1:
                # Boost score if keyword appears early (more likely primary intent).
                positional_boost = 1.0 + (0.5 * (1.0 - min(keyword_pos / 200, 1.0)))
                negated_weight_multiplier = 1.0

                # Check for negations *before* the keyword within the specified proximity
                search_window_start = max(0, keyword_pos - negation_proximity)
                search_window = prompt_lower[search_window_start:keyword_pos]

                for pattern, penalty in NEGATION_PATTERNS:
                    if re.search(pattern, search_window):
                        negated = True
                        score += weight * (1 - penalty)  # Reduce score by penalty
                        break  # Apply only one negation penalty per keyword match

                # Apply both positional boost and negation multiplier
                score += weight * positional_boost * negated_weight_multiplier

        # Additional heuristic boosts
        if (
            "code" in prompt_lower
            or "program" in prompt_lower
            or "script" in prompt_lower
        ) and (
            "analyze" in prompt_lower
            or "improve" in prompt_lower
            or "refactor" in prompt_lower
            or "evaluate" in prompt_lower
        ):
            score += 0.15

        if "project chimera" in prompt_lower and (
            "analyze" in prompt_lower or "improve" in prompt_lower
        ):
            score += 0.15

        # Explicit phrases that strongly indicate self-analysis
        explicit_phrases = [
            "analyze the entire Project Chimera codebase",
            "critically analyze the Project Chimera codebase",
            "perform self-analysis on the code",
            "evaluate my own implementation",
        ]
        if any(phrase in prompt_lower for phrase in explicit_phrases):
            score = max(score, 0.95)

        # Boost for multiple keyword matches with stricter requirements
        found_keywords_count = sum(1 for kw in SELF_ANALYSIS_KEYWORDS if kw in prompt_lower)
        if found_keywords_count > 1:
            score *= 1.1 ** (found_keywords_count - 1)
            score = min(score, 1.5)

        return score >= threshold

    @lru_cache(maxsize=256)
    def recommend_domain_from_keywords(self, user_prompt: str) -> Optional[str]:
        """
        Recommends a domain/framework based on keywords in the user prompt,
        incorporating negation awareness and proximity scoring.

        Args:
            user_prompt: The user's input prompt.

        Returns:
            The recommended domain name, or None if no strong match is found.
        """
        if not user_prompt or not self.domain_keywords:
            return None

        prompt_lower = user_prompt.lower()
        best_match = None
        highest_score = 0.0

        DEFAULT_KEYWORD_WEIGHT = 1.0
        NEGATION_PROXIMITY = 50

        for domain, keywords_list in self.domain_keywords.items():
            if not isinstance(keywords_list, list):
                continue

            score = 0.0
            for keyword in keywords_list:
                keyword_lower = keyword.lower()
                for match in re.finditer(
                    r"\b" + re.escape(keyword_lower) + r"\b", prompt_lower
                ):
                    keyword_start_pos = match.start()
                    negated = False
                    search_window_start = max(0, keyword_start_pos - NEGATION_PROXIMITY)
                    search_window = prompt_lower[search_window_start:keyword_start_pos]

                    for neg_pattern, penalty in NEGATION_PATTERNS:
                        if re.search(neg_pattern, search_window):
                            negated = True
                            score += DEFAULT_KEYWORD_WEIGHT * (1 - penalty)
                            break

                    if not negated:
                        score += DEFAULT_KEYWORD_WEIGHT

            if domain == "Software Engineering":
                if any(
                    term in prompt_lower
                    for term in ["implement", "endpoint", "api", "function"]
                ):
                    if "error handling" in prompt_lower or "validation" in prompt_lower:
                        score += 2.0

            if score > highest_score:
                highest_score = score
                best_match = domain

        if highest_score >= 1.0:
            return best_match

        return None

# --- NEW FUNCTION DEFINITION ---
@lru_cache(maxsize=256) # Cache results for identical prompts
def optimize_reasoning_prompt(prompt: str) -> str:
    """
    Optimizes a prompt for clarity, conciseness, and focus, potentially
    by removing redundant phrases, standardizing formatting, or emphasizing key instructions.

    Args:
        prompt: The original prompt string.

    Returns:
        An optimized prompt string.
    """
    if not prompt:
        return ""

    optimized_prompt = prompt.strip()

    # Basic cleanup: remove excessive whitespace between words and lines
    optimized_prompt = re.sub(r'\s+', ' ', optimized_prompt) # Replace multiple whitespaces with single space
    optimized_prompt = re.sub(r'\n\s*\n', '\n\n', optimized_prompt) # Normalize blank lines

    # Add more sophisticated prompt optimization logic here if needed, e.g.:
    # - Emphasizing critical instructions (e.g., JSON output format)
    # - Removing conversational filler
    # - Standardizing phrasing for common tasks

    # For now, a basic cleanup is sufficient to resolve the import error.
    logger.debug("Applied basic prompt optimization (cleanup).")
    return optimized_prompt

# --- END NEW FUNCTION DEFINITION ---