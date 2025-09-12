# src/constants.py

import re
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

# --- Constants for Prompt Analysis ---

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
NEGATION_PATTERNS: List[Tuple[str, float]] = [
    (r"(?i)\b(not|don\'t|do not|avoid|without|never|no)\b", 0.7),
    (r"(?i)\b(please do not|kindly avoid|do not intend to)\b", 0.9),
]

THRESHOLD = 0.75

# --- Constants for Persona Routing ---

# Optimized persona sequence for self-analysis prompts
SELF_ANALYSIS_PERSONA_SEQUENCE: List[str] = [
    "Self_Improvement_Analyst",  # Primary analyst first
    "Code_Architect",
    "Security_Auditor",
    "DevOps_Engineer",
    "Test_Engineer",
    "Constructive_Critic",
    "Impartial_Arbitrator",
    "Devils_Advocate",  # Devils Advocate is often useful for critical self-reflection
]

# --- SHARED CONSTANTS FOR LLM OUTPUT FORMATTING ---

# Common JSON output instructions that are repeated across multiple persona prompts.
# This constant centralizes these instructions to avoid redundancy and ensure consistency.
SHARED_JSON_INSTRUCTIONS: str = """
---
**CRITICAL JSON OUTPUT INSTRUCTIONS: ABSOLUTELY MUST BE FOLLOWED**
1. MUST BE A SINGLE, VALID JSON OBJECT. NO ARRAYS.
2. NO NUMBERED ARRAY ELEMENTS (e.g., "0:{...}" is INVALID).
3. ABSOLUTELY NO CONVERSATIONAL TEXT, MARKDOWN FENCES (```json, ```python, ```), OR EXPLANATIONS OUTSIDE THE JSON OBJECT.
4. STRICTLY ADHERE TO THE PROVIDED SCHEMA.
5. USE ONLY DOUBLE QUOTES for all keys and string values.
6. ENSURE COMMAS separate all properties in objects and elements in arrays.
7. DO NOT include trailing commas.
8. INCLUDE `malformed_blocks` FIELD (even if empty).
---

**CRITICAL DIFF FORMAT INSTRUCTION:** For any `CODE_CHANGES_SUGGESTED` with `ACTION: "MODIFY"`, the `DIFF_CONTENT` field MUST be a valid [Unified Diff Format](https://www.gnu.org/software/diffutils/manual/html_node/Unified-Format.html). It MUST start with `--- a/path/to/file` and `+++ b/path/to/file` headers, followed by lines starting with `+`, `-`, or ` ` (space). Example:
```diff
--- a/src/example.py
+++ b/src/example.py
@@ -1,3 +1,4 @@
 def old_func():
-    print("old")
+    print("new")
+    print("added")
```
**CRITICAL REMOVE FORMAT INSTRUCTION:** For any `CODE_CHANGES_SUGGESTED` with `ACTION: "REMOVE"`, you MUST provide the `LINES` field containing a non-empty list of the exact lines to be removed. `FULL_CONTENT` and `DIFF_CONTENT` MUST be null.
"""

# --- Other potential constants could be added here ---
# e.g., default file paths, common error messages, etc.
