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
    "critique your own output": 0.9
}

# Threshold for determining if a prompt is considered self-analysis.
# This value might require tuning based on empirical testing.
THRESHOLD = 0.85

def is_self_analysis_prompt(prompt: str) -> bool:
    """Checks if weighted keyword score meets threshold for self-analysis."""
    prompt_lower = prompt.lower()
    score = 0.0
    
    # Calculate score based on keyword presence and their weights
    for keyword, weight in SELF_ANALYSIS_KEYWORDS.items():
        if keyword in prompt_lower:
            score += weight
            
    # Apply context bonuses for combinations of keywords
    # This heuristic can be expanded for more nuanced detection.
    if "code" in prompt_lower and ("analyze" in prompt_lower or "improve" in prompt_lower):
        score *= 1.2
        
    # Return True if the calculated score meets or exceeds the threshold
    return score >= THRESHOLD

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