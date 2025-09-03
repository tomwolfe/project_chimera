# src/constants.py

import re
import logging
from typing import Dict, Any, List

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
    "critique my code": 0.77,
}

# Keywords and patterns for negation detection, used to reduce the score of self-analysis prompts.
NEGATION_PATTERNS = [
    (r"(?i)\b(not|don\'t|do not|avoid|without|never|no)\b", 0.7),
    (r"(?i)\b(please do not|kindly avoid|do not intend to)\b", 0.9),
]

THRESHOLD = 0.75

# Optimized persona sequence for self-analysis prompts
SELF_ANALYSIS_PERSONA_SEQUENCE = [
    "Code_Architect",
    "Security_Auditor",
    "Test_Engineer",
    "DevOps_Engineer",
    "Constructive_Critic",
    "Impartial_Arbitrator",
    "Devils_Advocate",
]