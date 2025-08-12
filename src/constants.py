# src/constants.py

from functools import lru_cache
import re
import logging
from typing import Dict, Any, List # Added List for type hinting

logger = logging.getLogger(__name__)

# Centralized keywords with weights for self-analysis prompt detection.
# Higher weights indicate stronger indicators.
SELF_ANALYSIS_KEYWORDS = {
    "analyze the entire Project Chimera codebase": 0.95,
    "critically analyze": 0.88,
    "self-analysis": 0.85,
    "improve code quality": 0.75,
    "refactor this code": 0.70,
    "evaluate my own code": 0.82,
    "identify code improvements": 0.78,
    "suggest code enhancements": 0.73,
    "analyze my own implementation": 0.80,
    "critique my code": 0.77
}

# Keywords and patterns for negation detection, used to reduce the score of self-analysis prompts.
NEGATION_PATTERNS = [
    (r'\b(not|don\'t|do not|avoid|without|never|no)\b', 0.7),
    (r'\b(please do not|kindly avoid|do not intend to)\b', 0.9)
]

THRESHOLD = 0.75

@lru_cache(maxsize=256)
def is_self_analysis_prompt(
    prompt: str,
    threshold: float = THRESHOLD,
    negation_proximity: int = 100 # Default proximity for negation check
) -> bool:
    """
    Checks if weighted keyword score meets threshold for self-analysis,
    incorporating negation handling and context-aware weighting.
    """
    prompt_lower = prompt.lower()
    score = 0.0
    
    for keyword, weight in SELF_ANALYSIS_KEYWORDS.items():
        keyword_pos = prompt_lower.find(keyword)
        if keyword_pos != -1:
            negated_weight_multiplier = 1.0
            
            # Check for negations *before* the keyword within the specified proximity
            search_window_end = keyword_pos
            search_window = prompt_lower[:search_window_end]
            
            # Find the closest negation pattern before the keyword
            closest_negation_end = -1
            for pattern, penalty in NEGATION_PATTERNS:
                for neg_match in re.finditer(pattern, search_window):
                    # Ensure the negation is within the proximity window
                    distance = keyword_pos - neg_match.end()
                    if distance >= 0 and distance <= negation_proximity:
                        if neg_match.end() > closest_negation_end: # Take the closest negation
                            closest_negation_end = neg_match.end()
                            negated_weight_multiplier = penalty # Apply the penalty
            
            score += weight * negated_weight_multiplier
    
    # Additional heuristic boosts
    if ("code" in prompt_lower or "program" in prompt_lower or "script" in prompt_lower) and \
       ("analyze" in prompt_lower or "improve" in prompt_lower or "refactor" in prompt_lower or "evaluate" in prompt_lower):
        score += 0.15
    
    if "project chimera" in prompt_lower and ("analyze" in prompt_lower or "improve" in prompt_lower):
        score += 0.10
    
    # Explicit phrases that strongly indicate self-analysis
    explicit_phrases = [
        "analyze the entire Project Chimera codebase",
        "critically analyze the Project Chimera codebase",
        "perform self-analysis on the code",
        "evaluate my own implementation"
    ]
    if any(phrase in prompt_lower for phrase in explicit_phrases):
        score = max(score, 0.92) # Ensure these phrases trigger self-analysis
    
    return score >= threshold

# Optimized persona sequence for self-analysis prompts
SELF_ANALYSIS_PERSONA_SEQUENCE = [
    "Code_Architect",
    "Security_Auditor",
    "Test_Engineer",
    "DevOps_Engineer",
    "Constructive_Critic",
    "Impartial_Arbitrator",
    "Devils_Advocate"
]