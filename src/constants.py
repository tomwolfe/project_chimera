# src/constants.py

# Centralized keywords for self-analysis prompt detection.
SELF_ANALYSIS_KEYWORDS = [
    "chimera", "your code", "self-analysis", "codebase", "refactor this code",
    "improve your logic", "analyze your performance", "optimize this process",
    # FIX: Expanded keyword set for comprehensive detection
    "critique your own output", "analyze the entire Project Chimera codebase",
    "system analysis", "self-improvement", "your reasoning"
]

def is_self_analysis_prompt(prompt: str) -> bool:
   """Checks if a prompt contains keywords indicating self-analysis."""
   prompt_lower = prompt.lower()
   return any(keyword in prompt_lower for keyword in SELF_ANALYSIS_KEYWORDS)

SELF_ANALYSIS_PERSONA_SEQUENCE = [
    "Context_Aware_Assistant",
    "Code_Architect",
    "Security_Auditor",
    "Constructive_Critic",
    "Test_Engineer",
    "DevOps_Engineer",
    "Impartial_Arbitrator"
]