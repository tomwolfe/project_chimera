# src/constants.py

from functools import lru_cache # Import lru_cache for caching

# Centralized keywords with weights for self-analysis prompt detection.
# Higher weights indicate stronger indicators.
SELF_ANALYSIS_KEYWORDS = {
    "chimera": 0.9,
    "codebase": 0.85,
    "self-analysis": 0.95,
    "analyze the entire Project Chimera codebase": 1.0,
    "refactor this code": 0.8,
    "improve your logic": 0.75,
    "system analysis": 0.8,
    "self-improvement": 0.7,
    "your reasoning": 0.65,
    "critique your own output": 0.9,
    "evaluate my own code": 0.9,
    "review this project": 0.7,
    "assess my performance": 0.75
}

# Keywords and patterns for negation detection, used to reduce the score of self-analysis prompts.
NEGATION_PATTERNS = ["don't", "do not", "avoid", "without", "not "]

# Threshold for determining if a prompt is considered self-analysis.
# This value might require tuning based on empirical testing.
THRESHOLD = 0.85

# Cache for the is_self_analysis_prompt function to improve performance
# when the same prompts are evaluated multiple times.
@lru_cache(maxsize=128) # Cache up to 128 unique prompts
def is_self_analysis_prompt(
    prompt: str,
    threshold: float = THRESHOLD,
    negation_proximity: int = 20 # How close negation needs to be to affect score
) -> bool:
    """
    Checks if weighted keyword score meets threshold for self-analysis,
    incorporating negation handling and context-aware weighting.
    
    Args:
        prompt: The user's input prompt string.
        threshold: The minimum score required to classify as self-analysis.
        negation_proximity: The character distance within which negation affects keyword weight.
        
    Returns:
        True if the prompt is classified as self-analysis, False otherwise.
    """
    prompt_lower = prompt.lower()
    score = 0.0
    
    # Calculate base score from keywords
    for keyword, weight in SELF_ANALYSIS_KEYWORDS.items():
        if keyword in prompt_lower:
            # Check for nearby negation patterns
            negated_weight_multiplier = 1.0
            if any(pattern in prompt_lower for pattern in NEGATION_PATTERNS):
                # Find the closest negation pattern to the keyword
                closest_neg_dist = float('inf')
                for neg_pattern in NEGATION_PATTERNS:
                    neg_idx = prompt_lower.find(neg_pattern)
                    kw_idx = prompt_lower.find(keyword)
                    if neg_idx != -1 and kw_idx != -1:
                        dist = abs(neg_idx - kw_idx)
                        if dist < closest_neg_dist:
                            closest_neg_dist = dist
                
                # Apply penalty if negation is within proximity
                if closest_neg_dist < negation_proximity:
                    negated_weight_multiplier = 0.3 # Reduce weight significantly if negated
            
            score += weight * negated_weight_multiplier
    
    # Apply context bonuses for meaningful keyword combinations
    # These bonuses help disambiguate general prompts from specific self-analysis requests.
    if "code" in prompt_lower and ("analyze" in prompt_lower or "improve" in prompt_lower or "refactor" in prompt_lower):
        score += 0.2 # Boost for code-related analysis/improvement prompts
    if "project chimera" in prompt_lower and ("analyze" in prompt_lower or "improve" in prompt_lower):
        score += 0.15 # Boost for prompts explicitly mentioning the project name for analysis
    
    # Explicit phrases that should strongly indicate self-analysis
    explicit_phrases = [
        "analyze the entire Project Chimera codebase",
        "critically analyze the codebase",
        "perform self-analysis",
        "evaluate my own code"
    ]
    if any(phrase in prompt_lower for phrase in explicit_phrases):
        score = max(score, 1.0)  # Ensure explicit requests always trigger if score is below 1.0
    
    # Return True if the calculated score meets or exceeds the threshold
    return score >= threshold

# Standardized persona sequence for self-analysis prompts
SELF_ANALYSIS_PERSONA_SEQUENCE = [
    "Context_Aware_Assistant",
    "Code_Architect",
    "Security_Auditor",
    "Constructive_Critic",
    "Test_Engineer",
    "DevOps_Engineer",
    "Impartial_Arbitrator"
]