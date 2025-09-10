# tests/integration/test_reasoning_engine.py
import pytest
import os
from unittest.mock import patch, MagicMock

# Assuming these imports are correct based on your project structure
# Adjust if your core.py or llm_provider.py are in different locations
from core import (
    SocraticDebate,
)  # SocraticDebate is in the project root, so direct import
from src.llm_provider import (
    GeminiProvider,
)  # Assuming GeminiProvider is in src/llm_provider.py
from src.persona_manager import PersonaManager # Import PersonaManager
from src.token_tracker import TokenUsageTracker # Import TokenUsageTracker


# Fixture to provide a real LLM client for integration tests.
# Requires TEST_LLM_API_KEY environment variable to be set.
@pytest.fixture
def real_llm_client():
    api_key = os.environ.get("TEST_LLM_API_KEY")
    if not api_key:
        pytest.skip(
            "TEST_LLM_API_KEY environment variable not set. Skipping real LLM integration tests."
        )

    try:
        return GeminiProvider(api_key=api_key, model_name="gemini-2.5-flash-lite")
    except Exception as e:
        pytest.fail(f"Failed to initialize real LLM client: {e}")


@pytest.mark.integration
def test_reasoning_engine_integration(real_llm_client):
    """
    Performs an end-to-end integration test of the SocraticDebate engine
    using a real LLM client.
    """
    api_key = os.environ.get("TEST_LLM_API_KEY")
    if not api_key:
        pytest.skip(
            "TEST_LLM_API_KEY environment variable not set for SocraticDebate init."
        )

    user_input = "What is the capital of France?"
    context = {"country": "France"}

    # Mock DOMAIN_KEYWORDS for PersonaManager initialization
    mock_domain_keywords = {
        "General": ["general", "question", "answer", "capital", "country"],
        "Software Engineering": ["code", "python", "implement"],
    }
    
    # Initialize TokenUsageTracker
    token_tracker_instance = TokenUsageTracker(budget=100000) # Provide a reasonable budget

    # Initialize PersonaManager with mock domain keywords and token tracker
    persona_manager_instance = PersonaManager(mock_domain_keywords, token_tracker=token_tracker_instance)

    try:
        engine = SocraticDebate(
            initial_prompt=user_input,
            api_key=api_key,
            model_name="gemini-2.5-flash-lite",  # Use a light model for tests
            domain="General",  # Use 'General' for simple questions
            persona_manager=persona_manager_instance,  # Pass the persona manager
            structured_codebase_context={}, # NEW: Add structured_codebase_context
            raw_file_contents={}, # NEW: Add raw_file_contents
            token_tracker=token_tracker_instance # Pass the token tracker
        )

        result, intermediate_steps = engine.run_debate()

        assert isinstance(result, dict)
        assert (
            "general_output" in result
        )
        assert (
            "Paris" in result["general_output"] or "paris" in result["general_output"]
        )

        print(f"Integration test successful. Result: {result['general_output']}")
        print(f"Total tokens used: {intermediate_steps.get('Total_Tokens_Used', 0)}")
        print(
            f"Estimated cost: ${intermediate_steps.get('Total_Estimated_Cost_USD', 0.0):.6f}"
        )

    except Exception as e:
        pytest.fail(f"Integration test failed with exception: {e}")