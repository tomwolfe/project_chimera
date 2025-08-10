# src/constants.py

from functools import lru_cache # Import lru_cache for caching

# Centralized keywords with weights for self-analysis prompt detection.
# Higher weights indicate stronger indicators.
SELF_ANALYSIS_KEYWORDS = {
    "analyze the entire Project Chimera codebase": 1.0,
    "critically analyze": 0.95,
    "self-analysis": 0.93,
    "improve code quality": 0.88,
    "refactor this code": 0.85,
    "evaluate my own code": 0.92
}

# Keywords and patterns for negation detection, used to reduce the score of self-analysis prompts.
# Patterns are tuples: (regex_pattern, penalty_multiplier)
NEGATION_PATTERNS = [
    (r'\b(not|don\'t|do not|avoid|without)\b.*?\b(code|analyze|refactor|improve)\b', 0.7),
    (r'\b(code|analyze|refactor|improve)\b.*?\b(not|don\'t|do not|avoid|without)\b', 0.7),
    (r'\b(please do not|kindly avoid)\b', 0.9)
]

# Threshold for determining if a prompt is considered self-analysis.
# This value might require tuning based on empirical testing.
THRESHOLD = 0.82  # Slightly lowered due to improved precision

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
            negated_weight_multiplier = 1.0
            # Check for negation patterns
            for pattern, penalty in NEGATION_PATTERNS:
                # Use re.search to find matches for the negation pattern
                neg_match = re.search(pattern, prompt_lower)
                if neg_match:
                    # Check if the keyword is within the proximity of the negation match
                    keyword_match = re.search(re.escape(keyword), prompt_lower)
                    if keyword_match:
                        # Calculate distance between the start of the negation match and the keyword match
                        distance = abs(neg_match.start() - keyword_match.start())
                        if distance < negation_proximity:
                            negated_weight_multiplier = min(negated_weight_multiplier, penalty) # Apply the penalty
                            break # Apply only the strongest penalty if multiple negations apply
            
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