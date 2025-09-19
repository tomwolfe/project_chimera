# tests/test_prompt_engineering.py

import pytest
from src.utils.prompting.prompt_engineering import format_prompt
from src.persona_manager import PersonaManager  # Needed for mocking in session_manager


# Mock app_config and EXAMPLE_PROMPTS for session_manager initialization
@pytest.fixture
def mock_app_config():
    return {
        "max_tokens_limit": 2000000,
        "context_token_budget_ratio": 0.25,
        "domain_keywords": {"General": ["general"], "Software Engineering": ["code"]},
        "example_prompts": {
            "Coding & Implementation": {
                "Implement Python API Endpoint": {
                    "prompt": "Implement a new FastAPI endpoint.",
                    "description": "Generate an API endpoint.",
                    "framework_hint": "Software Engineering",
                }
            }
        },
    }


def test_format_prompt_basic():
    """Test format_prompt with basic variable substitution."""
    template = "Hello, {name}!"
    kwargs = {"name": "World"}
    result = format_prompt(template, **kwargs)
    assert result == "Hello, World!"


def test_format_prompt_with_codebase_context_self_analysis():
    """Test format_prompt with codebase context for self-analysis."""
    template = "Analyze this: {issue}"
    codebase_context = {
        "file_structure": {"critical_files_preview": {"file1.py": "def func(): pass"}}
    }
    kwargs = {"issue": "bug"}
    result = format_prompt(
        template, codebase_context=codebase_context, is_self_analysis=True, **kwargs
    )
    assert "CODEBASE CONTEXT" in result
    assert "file1.py" in result
    assert "bug" in result


def test_format_prompt_missing_key():
    """Test format_prompt handles missing keys gracefully."""
    template = "Hello, {name}!"
    kwargs = {"age": 30}  # Missing 'name'
    result = format_prompt(
        template, **kwargs
    )  # The function logs a warning, but doesn't modify the returned string with the warning.
    assert result == "Hello, {name}!"  # The placeholder should remain if not formatted
