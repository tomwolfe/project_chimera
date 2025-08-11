# src/constants.py

from functools import lru_cache # Import lru_cache for caching

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
    "suggest code enhancements": 0.73
}

# Keywords and patterns for negation detection, used to reduce the score of self-analysis prompts.
# Patterns are tuples: (regex_pattern, penalty_multiplier)
# MODIFIED: Added more specific patterns and directional logic hints.
NEGATION_PATTERNS = [
    # Negation appearing before the keyword, with proximity check
    (r'\b(not|don\'t|do not|avoid|without|never|no)\b', 0.7),
    # Negation appearing after the keyword (less common, but possible)
    (r'\b(not|don\'t|do not|avoid|without|never|no)\b', 0.7),
    # Phrases that strongly negate analysis intent
    (r'\b(please do not|kindly avoid|do not intend to)\b', 0.9)
]

# Threshold for determining if a prompt is considered self-analysis.
THRESHOLD = 0.75

# Cache for the is_self_analysis_prompt function to improve performance
# when the same prompts are evaluated multiple times.
@lru_cache(maxsize=256) # Cache up to 256 unique prompts
def is_self_analysis_prompt(
    prompt: str,
    threshold: float = THRESHOLD,
    negation_proximity: int = 100 # Increased proximity for better context
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
        keyword_pos = prompt_lower.find(keyword)
        if keyword_pos != -1:
            negated_weight_multiplier = 1.0
            
            # MODIFIED: Directional negation analysis
            # Check for negation patterns appearing BEFORE the keyword
            for pattern, penalty in NEGATION_PATTERNS:
                # Iterate through all matches of the pattern in the prompt up to the keyword's position
                for neg_match in re.finditer(pattern, prompt_lower[:keyword_pos + len(keyword)]):
                    # Ensure the negation match ends before or at the start of the keyword
                    if neg_match.end() <= keyword_pos:
                        distance = keyword_pos - neg_match.end()
                        if distance < negation_proximity:
                            negated_weight_multiplier = min(negated_weight_multiplier, penalty)
                            # Apply the strongest penalty found before the keyword and break
                            break 
            
            # Check for negation patterns appearing AFTER the keyword (less common, but possible)
            # This part is less critical for the specific "do not analyze" case but adds robustness.
            for pattern, penalty in NEGATION_PATTERNS:
                # Iterate through all matches of the pattern starting from the keyword's position
                for neg_match in re.finditer(pattern, prompt_lower[keyword_pos:]):
                    # Calculate the actual start position of the negation in the original string
                    actual_neg_pos = keyword_pos + neg_match.start()
                    # Ensure the negation starts after the keyword ends
                    if actual_neg_pos >= keyword_pos + len(keyword):
                        distance = actual_neg_pos - (keyword_pos + len(keyword))
                        if distance < negation_proximity:
                            negated_weight_multiplier = min(negated_weight_multiplier, penalty)
                            # Apply the strongest penalty found after the keyword and break
                            break
            
            score += weight * negated_weight_multiplier
    
    # Apply context bonuses for meaningful keyword combinations
    if "code" in prompt_lower and ("analyze" in prompt_lower or "improve" in prompt_lower or "refactor" in prompt_lower):
        score += 0.2
    if "project chimera" in prompt_lower and ("analyze" in prompt_lower or "improve" in prompt_lower):
        score += 0.15
    
    # Explicit phrases that should strongly indicate self-analysis
    explicit_phrases = [
        "analyze the entire Project Chimera codebase",
        "critically analyze the Project Chimera codebase",
        "perform self-analysis on the code",
        "evaluate my own implementation"
    ]
    if any(phrase in prompt_lower for phrase in explicit_phrases):
        score = max(score, 0.92)
    
    return score >= threshold

# Optimized persona sequence for self-analysis prompts
# KNOWLEDGE BASE REFERENCE: Lines 120-150 (persona definitions and focus areas)
SELF_ANALYSIS_PERSONA_SEQUENCE = [
    "Code_Architect",
    "Security_Auditor",
    "Test_Engineer", # Moved earlier per knowledge base focus on code quality
    "Constructive_Critic",
    "DevOps_Engineer",
    "Impartial_Arbitrator"
]