import pytest
from ai_core.self_improvement_loop import SelfImprovementLoop
from src.llm_provider import LLMProvider
from src.utils.prompt_engineering import create_self_improvement_prompt


# Mock LLMProvider for testing the loop's orchestration
class MockLLMProvider:
    def __init__(self):
        self.call_count = 0

    def generate_content(self, prompt):
        self.call_count += 1
        # Simulate different responses based on turn or prompt content
        if "Critically analyze" in prompt and "code_snippet" in prompt:
            # Simulate a valid JSON output for the first turn (analysis)
            if self.call_count == 1:
                return """
                {
                  "ANALYSIS_SUMMARY": "Initial analysis complete.",
                  "IMPACTFUL_SUGGESTIONS": [
                    {
                      "AREA": "Security",
                      "PROBLEM": "Potential SQL Injection in database_operations.py",
                      "PROPOSED_SOLUTION": "Use parameterized queries.",
                      "EXPECTED_IMPACT": "Prevents SQL injection.",
                      "CODE_CHANGES_SUGGESTED": [
                        {
                          "FILE_PATH": "src/database_operations.py",
                          "ACTION": "MODIFY",
                          "DIFF_CONTENT": "--- a/src/database_operations.py\\n+++ b/src/database_operations.py\\n@@ -10,7 +10,7 @@\\n def get_user_data(user_id):\\n     # Vulnerable to SQL injection\\n     query = f\\"SELECT * FROM users WHERE id = {user_id}\\"\\n     return execute_query(query)\\n+++ b/src/database_operations.py\\n@@ -10,7 +10,7 @@\\n def get_user_data(user_id):\\n     # Safe parameterized query\\n     query = \\"SELECT * FROM users WHERE id = ?\\"\\n     return execute_query(query, (user_id,))\\n"
                        }
                      ]
                    }
                  ]
                }
                """
            # Simulate conflict resolution response
            elif self.call_count == 2:
                return """
                {
                  "CONFLICT_RESOLUTION": "Prioritizing security fix for SQL injection as it addresses critical vulnerability.",
                  "UNRESOLVED_CONFLICT": null
                }
                """
            # Simulate final answer with code changes
            elif self.call_count == 3:
                return """
                {
                  "COMMIT_MESSAGE": "Fix SQL injection vulnerability",
                  "RATIONALE": "Addressed critical SQL injection vulnerability in database operations",
                  "CODE_CHANGES": [
                    {
                      "FILE_PATH": "src/database_operations.py",
                      "ACTION": "MODIFY",
                      "DIFF_CONTENT": "--- a/src/database_operations.py\\n+++ b/src/database_operations.py\\n@@ -10,7 +10,7 @@\\n def get_user_data(user_id):\\n     # Vulnerable to SQL injection\\n     query = f\\"SELECT * FROM users WHERE id = {user_id}\\"\\n     return execute_query(query)\\n+++ b/src/database_operations.py\\n@@ -10,7 +10,7 @@\\n def get_user_data(user_id):\\n     # Safe parameterized query\\n     query = \\"SELECT * FROM users WHERE id = ?\\"\\n     return execute_query(query, (user_id,))\\n"
                    }
                  ],
                  "malformed_blocks": []
                }
                """
        return "Mock response"


def test_self_improvement_loop_initialization():
    """Test that SelfImprovementLoop initializes correctly."""
    loop = SelfImprovementLoop()
    assert loop is not None
    assert hasattr(loop, "intermediate_steps")
    assert isinstance(loop.intermediate_steps, dict)


def test_self_improvement_loop_full_cycle(monkeypatch):
    """Test the complete self-improvement loop cycle."""
    # Replace the actual LLM provider with our mock
    mock_provider = MockLLMProvider()
    monkeypatch.setattr(
        "ai_core.self_improvement_loop.LLMProvider", lambda: mock_provider
    )

    loop = SelfImprovementLoop()
    result = loop.run_self_improvement_cycle()

    # Verify the loop completed all stages
    assert "ANALYSIS_SUMMARY" in result
    assert "IMPACTFUL_SUGGESTIONS" in result
    assert "CONFLICT_RESOLUTION" in result
    assert "FINAL_ANSWER" in result

    # Verify token usage metrics were recorded
    assert "token_usage" in loop.intermediate_steps
    assert loop.intermediate_steps["token_usage"]["total_tokens"] > 0

    # Verify code changes were properly formatted
    final_answer = result["FINAL_ANSWER"]
    assert "COMMIT_MESSAGE" in final_answer
    assert "CODE_CHANGES" in final_answer
    assert isinstance(final_answer["CODE_CHANGES"], list)
    if final_answer["CODE_CHANGES"]:
        assert "FILE_PATH" in final_answer["CODE_CHANGES"][0]
        assert "ACTION" in final_answer["CODE_CHANGES"][0]


def test_self_improvement_loop_error_handling():
    """Test error handling in the self-improvement loop."""

    # Create a mock provider that raises an exception
    class ErrorLLMProvider:
        def generate_content(self, prompt):
            raise Exception("LLM API Error")

    # Patch the LLMProvider to use our error-raising mock
    from unittest.mock import MagicMock

    MagicMock.side_effect = Exception("LLM API Error")

    loop = SelfImprovementLoop()
    try:
        loop.run_self_improvement_cycle()
        assert False, "Expected exception was not raised"
    except Exception as e:
        assert "LLM API Error" in str(e)
