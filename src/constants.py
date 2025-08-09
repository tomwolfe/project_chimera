# src/constants.py

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

# Threshold for determining if a prompt is considered self-analysis.
# This value might require tuning based on empirical testing.
THRESHOLD = 0.85

def is_self_analysis_prompt(prompt: str) -> bool:
    """
    Checks if weighted keyword score meets threshold for self-analysis,
    incorporating negation handling and context-aware weighting.
    """
    prompt_lower = prompt.lower()
    score = 0.0
    
    # Check for negation patterns that should suppress self-analysis
    negation_patterns = ["don't", "do not", "avoid", "without", "not "]
    # Determine the effective threshold based on presence of negation
    base_threshold = THRESHOLD
    if any(pattern in prompt_lower for pattern in negation_patterns):
        # If negation is present, we need a stronger signal to override it.
        # Increase the threshold significantly.
        base_threshold = THRESHOLD * 1.5
    
    # Calculate score based on keyword presence and their weights
    for keyword, weight in SELF_ANALYSIS_KEYWORDS.items():
        if keyword in prompt_lower:
            # Reduce weight if negation pattern appears nearby the keyword
            negation_proximity = 20  # characters: how close negation needs to be to affect weight
            negated = False
            for neg_pattern in negation_patterns:
                neg_idx = prompt_lower.find(neg_pattern)
                kw_idx = prompt_lower.find(keyword)
                if neg_idx != -1 and kw_idx != -1 and abs(neg_idx - kw_idx) < negation_proximity:
                    # Reduce weight significantly if negation is nearby
                    weight *= 0.3  
                    negated = True
                    break # Found a nearby negation for this keyword
            
            score += weight
            
    # Apply context bonuses for meaningful keyword combinations
    # These bonuses help disambiguate general prompts from specific self-analysis requests.
    if "code" in prompt_lower and ("analyze" in prompt_lower or "improve" in prompt_lower or "refactor" in prompt_lower):
        score += 0.2 # Boost for code-related analysis/improvement prompts
    if "project chimera" in prompt_lower and ("analyze" in prompt_lower or "improve" in prompt_lower):
        score += 0.15 # Boost for prompts explicitly mentioning the project name for analysis
    
    # Check for explicit self-analysis phrases that should guarantee triggering
    explicit_phrases = [
        "analyze the entire Project Chimera codebase",
        "critically analyze the codebase",
        "perform self-analysis",
        "evaluate my own code"
    ]
    if any(phrase in prompt_lower for phrase in explicit_phrases):
        score = max(score, 1.0)  # Ensure explicit requests always trigger if score is below 1.0
    
    # Return True if the calculated score meets or exceeds the effective threshold
    return score >= base_threshold

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