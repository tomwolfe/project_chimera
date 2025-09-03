# src/utils/domain_recommender.py
from typing import Dict, Any, Optional, List
import re # Import re for regex

# Import NEGATION_PATTERNS and define proximity
from src.constants import NEGATION_PATTERNS


def recommend_domain_from_keywords(
    user_prompt: str, domain_keywords: Dict[str, List[str]]
) -> Optional[str]:
    """
    Recommends a domain/framework based on keywords in the user prompt,
    incorporating negation awareness and proximity scoring.

    Args:
        user_prompt: The user's input prompt.
        domain_keywords: A dictionary mapping domain names to lists of keywords.

    Returns:
        The recommended domain name, or None if no strong match is found.
    """
    if not user_prompt or not domain_keywords:
        return None

    prompt_lower = user_prompt.lower()
    best_match = None
    highest_score = 0.0 # Use float for score

    DEFAULT_KEYWORD_WEIGHT = 1.0
    NEGATION_PROXIMITY = 50 # Define proximity for negation check (e.g., 50 characters)

    for domain, keywords_list in domain_keywords.items():
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
                # keyword_end_pos = match.end() # Not used in this logic

                # Check for negations *before* the keyword within proximity
                negated = False
                search_window_start = max(0, keyword_start_pos - NEGATION_PROXIMITY)
                search_window = prompt_lower[search_window_start:keyword_start_pos]

                for neg_pattern, penalty in NEGATION_PATTERNS:
                    if re.search(neg_pattern, search_window):
                        negated = True
                        score += DEFAULT_KEYWORD_WEIGHT * (
                            1 - penalty
                        ) # Reduce score by penalty
                        break # Apply only one negation penalty per keyword match

                if not negated:
                    score += DEFAULT_KEYWORD_WEIGHT # Add full weight if not negated

        # Additional context boost for technical terms in code contexts (from LLM's suggestion)
        if domain == "Software Engineering":
            if any(
                term in prompt_lower
                for term in ["implement", "endpoint", "api", "function"]
            ):
                if "error handling" in prompt_lower or "validation" in prompt_lower:
                    score += (
                        2.0 # Significant boost for complete implementation requests
                    )

        if score > highest_score:
            highest_score = score
            best_match = domain

    # A higher threshold might be needed for more confident recommendations
    # Adjust this based on desired strictness. 1.0 means at least one non-negated keyword match.
    if highest_score >= 1.0:
        return best_match

    return None