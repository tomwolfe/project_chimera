# src/utils/prompt_analyzer.py
import re
from typing import Dict, Any, List, Optional
from functools import lru_cache

# Assuming SELF_ANALYSIS_KEYWORDS, NEGATION_PATTERNS, and THRESHOLD are imported from src.constants
# These are global constants, so importing them directly is appropriate.
from src.constants import SELF_ANALYSIS_KEYWORDS, NEGATION_PATTERNS, THRESHOLD


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

        return {
            "complexity_score": complexity_score,
            "primary_domain": primary_domain,
            "domain_scores": domain_scores,
            "word_count": word_count,
            "sentence_count": sentence_count,
        }

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
                        negated_weight_multiplier = penalty  # Apply the penalty
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
            score += 0.10

        # Explicit phrases that strongly indicate self-analysis
        explicit_phrases = [
            "analyze the entire Project Chimera codebase",
            "critically analyze the Project Chimera codebase",
            "perform self-analysis on the code",
            "evaluate my own implementation",
        ]
        if any(phrase in prompt_lower for phrase in explicit_phrases):
            score = max(score, 0.92)  # Ensure these phrases trigger self-analysis

        # Boost for multiple keyword matches
        found_keywords_count = sum(
            1 for kw in SELF_ANALYSIS_KEYWORDS if kw in prompt_lower
        )
        if found_keywords_count > 1:
            score *= 1.1 ** (found_keywords_count - 1)
            score = min(score, 1.5)  # Cap the total score from this boost

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
        NEGATION_PROXIMITY = (
            50  # Define proximity for negation check (e.g., 50 characters)
        )

        for domain, keywords_list in self.domain_keywords.items():
            if not isinstance(keywords_list, list):
                continue

            score = 0.0
            for keyword in keywords_list:
                keyword_lower = keyword.lower()
                # Find all occurrences of the keyword
                for match in re.finditer(
                    r"\b" + re.escape(keyword_lower) + r"\b", prompt_lower
                ):
                    keyword_start_pos = match.start()

                    # Check for negations *before* the keyword within proximity
                    negated = False
                    search_window_start = max(0, keyword_start_pos - NEGATION_PROXIMITY)
                    search_window = prompt_lower[search_window_start:keyword_start_pos]

                    for neg_pattern, penalty in NEGATION_PATTERNS:
                        if re.search(neg_pattern, search_window):
                            negated = True
                            score += DEFAULT_KEYWORD_WEIGHT * (
                                1 - penalty
                            )  # Reduce score by penalty
                            break  # Apply only one negation penalty per keyword match

                    if not negated:
                        score += (
                            DEFAULT_KEYWORD_WEIGHT  # Add full weight if not negated
                        )

            # Additional context boost for technical terms in code contexts
            if domain == "Software Engineering":
                if any(
                    term in prompt_lower
                    for term in ["implement", "endpoint", "api", "function"]
                ):
                    if "error handling" in prompt_lower or "validation" in prompt_lower:
                        score += 2.0  # Significant boost for complete implementation requests

            if score > highest_score:
                highest_score = score
                best_match = domain

        # Special case: if prompt contains code block indicators but low score, boost SE
        if highest_score < 1.0 and (
            "```python" in prompt_lower
            or ".py" in prompt_lower
            or "function" in prompt_lower
        ):
            return "Software Engineering"

        # A higher threshold might be needed for more confident recommendations
        if highest_score >= 1.0:
            return best_match

        return None
