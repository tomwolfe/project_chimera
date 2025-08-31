# src/utils/prompt_engineering.py
import json
from typing import Dict, Any

# NEW IMPORTS for Pydantic validation
from pydantic import BaseModel, ValidationError
import logging

# Assuming LLMResponseModel is defined in src/models.py
from src.models import LLMResponseModel

logger = logging.getLogger(__name__)

def create_self_improvement_prompt(metrics, previous_analyses=None):
    """Enhanced prompt that guides more targeted, actionable self-analysis."""
    security_prioritization = (
        "When identifying security issues:\n"
        "1. Prioritize HIGH severity Bandit issues first (SQL injection, command injection, hardcoded secrets)\n"
        "2. Group similar issues together rather than listing individually\n"
        "3. Provide specific examples of the MOST critical 3-5 vulnerabilities"
    )
    
    token_optimization = (
        "For token efficiency:\n"
        "1. Analyze which personas consume disproportionate tokens\n"
        "2. Identify repetitive or redundant analysis patterns\n"
        "3. Suggest specific prompt truncation strategies for high-token personas"
    )
    
    testing_prioritization = (
        "For test coverage:\n"
        "1. Prioritize testing core logic (SocraticDebate, LLM interaction) before UI components\n"
        "2. Focus on areas with highest bug density per historical data\n"
        "3. Implement targeted smoke tests for critical paths first"
    )
    
    self_reflection = (
        "CRITICAL: Analyze your OWN self-improvement process:\n"
        "1. What aspects of previous self-improvement analyses were most/least effective?\n"
        "2. How can the self-analysis framework be enhanced to produce better recommendations?\n"
        "3. What metrics would best measure the effectiveness of self-improvement changes?"
    )
    
    prompt = f"""You are conducting a self-improvement analysis of Project Chimera itself.
    
Key focus areas with specific guidance:
{security_prioritization}
{token_optimization}
{testing_prioritization}
{self_reflection}

Current metrics: {metrics}
{'Previous analyses: ' + str(previous_analyses) if previous_analyses else ''}

Provide analysis with:
1. Specific, prioritized recommendations (top 3-5)
2. Concrete code examples with complete implementation details
3. Expected impact metrics for each change
4. How this improves the self-improvement process itself
"""
    return prompt

def parse_llm_response(response: str) -> Dict[str, Any]:
    """
    Parses and validates the LLM's raw response using Pydantic.
    This replaces the simple json.loads with robust schema validation.
    """
    try:
        # Attempt to parse the raw response using the defined LLMResponseModel
        # .dict() converts the Pydantic model instance back to a dictionary
        parsed_response = LLMResponseModel.model_validate_json(response).dict()
        return parsed_response
    except ValidationError as e:
        logger.error(f"LLM response validation failed: {e}. Raw response: {response[:500]}...")
        # Re-raise the validation error, or return a structured error response
        # For robustness, we'll re-raise, allowing upstream error handling to catch it.
        # The SocraticDebate's _execute_llm_turn already handles SchemaValidationError.
        raise ValueError(f"LLM response validation failed: {e}") from e
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from LLM response: {e}. Raw response: {response[:500]}...")
        raise ValueError(f"LLM response is not valid JSON: {e}") from e
    except Exception as e:
        logger.error(f"An unexpected error occurred during LLM response parsing: {e}. Raw response: {response[:500]}...")
        raise ValueError(f"Unexpected error parsing LLM response: {e}") from e