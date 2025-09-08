# src/utils/constants.py

# This constant defines the generic critical JSON output instructions
# to be dynamically injected into persona prompts that require structured JSON.
SHARED_JSON_INSTRUCTIONS = """
---
**CRITICAL JSON OUTPUT INSTRUCTIONS: ABSOLUTELY MUST BE FOLLOWED**
1. MUST BE A SINGLE, VALID JSON OBJECT. NO ARRAYS.
2. NO NUMBERED ARRAY ELEMENTS.
3. ABSOLUTELY NO CONVERSATIONAL TEXT, MARKDOWN FENCES, OR EXPLANATIONS OUTSIDE JSON.
4. STRICTLY ADHERE TO SCHEMA.
5. USE DOUBLE QUOTES.
6. ENSURE COMMAS. NO TRAILING COMMAS.
7. PROPER JSON ARRAY SYNTAX: `[{"key": "value"}, {"key": "value"}]`.
8. Include `malformed_blocks` field (even if empty).
---
"""

# Other constants can be added here as needed
