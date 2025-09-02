import pytest
from src.utils.prompt_engineering import create_self_improvement_prompt, parse_code_analysis_output

def test_create_self_improvement_prompt_basic():
    """Test the basic functionality of creating a self-improvement prompt."""
    persona_description = "An expert Python developer."
    code_snippet = "def hello_world():\n    print('Hello, world!')"
    prompt = create_self_improvement_prompt(persona_description, code_snippet)

    assert "persona: An expert Python developer." in prompt
    assert "code_snippet: def hello_world():\n    print('Hello, world!')" in prompt
    assert "Critically analyze the following Python code snippet" in prompt

def test_create_self_improvement_prompt_with_context():
    """Test prompt creation with additional context."""
    persona_description = "A security analyst."
    code_snippet = "import os\nuser_input = input('Enter path: ')\nos.system(f'ls {user_input}')"
    context = "Focus on potential security vulnerabilities."
    prompt = create_self_improvement_prompt(persona_description, code_snippet, context=context)

    assert "persona: A security analyst." in prompt
    assert "code_snippet: import os\nuser_input = input('Enter path: ')\nos.system(f'ls {user_input}')" in prompt
    assert "Critically analyze the following Python code snippet" in prompt
    assert "Focus on potential security vulnerabilities." in prompt

def test_parse_code_analysis_output_valid_json():
    """Test parsing of a valid JSON output."""
    valid_json = """
    {
        "COMMIT_MESSAGE": "Fix security issue",
        "RATIONALE": "Addressed SQL injection vulnerability",
        "CODE_CHANGES": [
            {
                "FILE_PATH": "src/database.py",
                "ACTION": "MODIFY",
                "FULL_CONTENT": "def get_user_data(user_id):\\n    query = \\"SELECT * FROM users WHERE id = ?\\"\\n    return execute_query(query, (user_id,))"
            }
        ],
        "malformed_blocks": []
    }
    """
    result = parse_code_analysis_output(valid_json)
    
    assert result["COMMIT_MESSAGE"] == "Fix security issue"
    assert len(result["CODE_CHANGES"]) == 1
    assert result["CODE_CHANGES"][0]["FILE_PATH"] == "src/database.py"

def test_parse_code_analysis_output_malformed():
    """Test handling of malformed JSON output."""
    malformed_json = "This is not JSON"
    result = parse_code_analysis_output(malformed_json)
    
    assert "COMMIT_MESSAGE" in result
    assert "Error" in result["COMMIT_MESSAGE"]
    assert len(result["malformed_blocks"]) > 0