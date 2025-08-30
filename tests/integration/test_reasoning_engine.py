# tests/integration/test_reasoning_engine.py
import pytest
import os  # Added for os.environ.get
from unittest.mock import patch, MagicMock

# Assuming these imports are correct based on your project structure
# Adjust if your core.py or llm_provider.py are in different locations
from core import SocraticDebate  # SocraticDebate is in the project root, so direct import
from src.llm_provider import GeminiProvider  # Assuming GeminiProvider is in src/llm_provider.py


# Fixture to provide a real LLM client for integration tests.
# Requires TEST_LLM_API_KEY environment variable to be set.
@pytest.fixture
def real_llm_client():
    api_key = os.environ.get("TEST_LLM_API_KEY")
    if not api_key:
        pytest.skip("TEST_LLM_API_KEY environment variable not set. Skipping real LLM integration tests.")

    # Replace with actual LLM client initialization pointing to a test endpoint
    # Ensure API keys and configurations are handled securely (e.g., env vars)
    try:
        # Assuming GeminiProvider is the actual client wrapper used by SocraticDebate
        # and it can be instantiated directly for testing purposes.
        return GeminiProvider(api_key=api_key, model_name="gemini-2.5-flash-lite")
    except Exception as e:
        pytest.fail(f"Failed to initialize real LLM client: {e}")


@pytest.mark.integration
def test_reasoning_engine_integration(real_llm_client):
    """
    Performs an end-to-end integration test of the SocraticDebate engine
    using a real LLM client.
    """
    # Initialize SocraticDebate with the real LLM client
    # Note: SocraticDebate expects the API key directly, not a pre-initialized client.
    # We need to adapt the test or the SocraticDebate constructor.
    # For this test, let's assume SocraticDebate's constructor can take a pre-configured provider.
    # If not, you'd pass the api_key directly to SocraticDebate and it would create its own provider.

    # Re-reading SocraticDebate's __init__: it takes api_key, not llm_provider instance.
    # So, we need to pass the API key directly.
    api_key = os.environ.get("TEST_LLM_API_KEY")
    if not api_key:
        pytest.skip("TEST_LLM_API_KEY environment variable not set for SocraticDebate init.")

    user_input = "What is the capital of France?"
    context = {"country": "France"}

    # Initialize SocraticDebate directly with the API key
    # Assuming 'General' domain and default personas for a simple query
    # You might need to adjust the domain and persona_manager initialization
    # if SocraticDebate requires a fully configured PersonaManager.

    # For simplicity, let's create a minimal PersonaManager for the test
    from src.persona_manager import PersonaManager
    from app import DOMAIN_KEYWORDS  # Assuming DOMAIN_KEYWORDS is accessible or can be mocked

    # Mock DOMAIN_KEYWORDS if app.py is not directly imported or if it's not a global constant
    # For now, let's assume it's accessible.

    # Minimal PersonaManager setup for testing
    # In a real test, you might mock this or load a specific test config
    mock_domain_keywords = {"General": ["general", "question", "answer"], "Software Engineering": ["code", "python"]}
    persona_manager_instance = PersonaManager(mock_domain_keywords)

    try:
        engine = SocraticDebate(
            initial_prompt=user_input,
            api_key=api_key,
            model_name="gemini-2.5-flash-lite",  # Use a light model for tests
            domain="General",  # Use 'General' for simple questions
            persona_manager=persona_manager_instance,  # Pass the persona manager
        )

        result, intermediate_steps = engine.run_debate()

        assert isinstance(result, dict)
        assert "general_output" in result  # Adjust based on expected output structure for General domain
        assert "Paris" in result["general_output"] or "paris" in result["general_output"]  # Basic check

        # Add checks for latency, error rates if possible (requires more advanced metrics collection)
        print(f"Integration test successful. Result: {result['general_output']}")
        print(f"Total tokens used: {intermediate_steps.get('Total_Tokens_Used', 0)}")
        print(f"Estimated cost: ${intermediate_steps.get('Total_Estimated_Cost_USD', 0.0):.6f}")

    except Exception as e:
        pytest.fail(f"Integration test failed with exception: {e}")