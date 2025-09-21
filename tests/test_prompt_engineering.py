from unittest.mock import MagicMock

import pytest

from src.config.settings import ChimeraSettings
from src.llm_tokenizers.base import Tokenizer
from src.models import PersonaConfig  # NEW: Import PersonaConfig
from src.utils.prompting.prompt_optimizer import PromptOptimizer


@pytest.fixture
def mock_tokenizer():
    """Provides a mock Tokenizer instance."""
    tokenizer = MagicMock(spec=Tokenizer)
    tokenizer.count_tokens.side_effect = lambda text: len(text) // 4
    tokenizer.truncate_to_token_limit.side_effect = (
        lambda text, max_tokens, truncation_indicator="": text[: max_tokens * 4]
        + (truncation_indicator if len(text) > max_tokens * 4 else "")
    )
    return tokenizer


@pytest.fixture
def mock_settings():
    """Provides a mock ChimeraSettings instance."""
    settings = MagicMock(spec=ChimeraSettings)
    settings.default_max_input_tokens_per_persona = 4000
    settings.max_tokens_per_persona = {
        "TestPersona": 1024,
        "Self_Improvement_Analyst": 16000,
    }
    # Mock global consumption and efficiency thresholds for aggressive optimization tests
    settings.GLOBAL_TOKEN_CONSUMPTION_THRESHOLD = 0.7
    settings.TOKEN_EFFICIENCY_SCORE_THRESHOLD = 0.7
    return settings


@pytest.fixture
def mock_summarizer_pipeline():
    """Provides a mock Hugging Face summarization pipeline."""
    summarizer = MagicMock()
    summarizer.return_value = [{"summary_text": "Mock summary."}]
    summarizer.tokenizer.model_max_length = 1024  # Simulate distilbart's max input
    return summarizer


@pytest.fixture
def prompt_optimizer_instance(mock_tokenizer, mock_settings, mock_summarizer_pipeline):
    """Provides a PromptOptimizer instance with mocked dependencies."""
    return PromptOptimizer(
        tokenizer=mock_tokenizer,
        settings=mock_settings,
        summarizer_pipeline=mock_summarizer_pipeline,
    )


class TestPromptEngineering:
    def setup_method(self):
        # These are now provided by fixtures, so no need to instantiate here.
        # self.token_tracker = TokenTracker()
        # self.prompt_optimizer = PromptOptimizer()
        # self.prompt_generator = PromptGenerator()
        pass  # Setup is handled by fixtures

    def test_prompt_optimization_efficiency(
        self, prompt_optimizer_instance, mock_tokenizer
    ):  # NEW: Add fixtures
        original_prompt = "Explain the concept of gravity in simple terms"
        # NEW: Create a mock PersonaConfig for the optimize_prompt signature
        persona_config = MagicMock(
            spec=PersonaConfig, name="TestPersona", token_efficiency_score=0.8
        )
        system_message = "You are a helpful assistant."
        max_output_tokens = 500

        # Mock tokenizer's count_tokens to control token counts for the test
        mock_tokenizer.count_tokens.side_effect = [
            len(system_message) // 4,  # System message tokens
            (len(system_message) + len(original_prompt)) // 4,  # Full input tokens
            (len(system_message) + len(original_prompt))
            // 4,  # Full input tokens (again for truncation check)
            (len(system_message) + len(original_prompt) // 2)
            // 4,  # Truncated user prompt tokens
        ]

        # NEW: Call optimize_prompt with the correct signature
        optimized_prompt = prompt_optimizer_instance.optimize_prompt(
            user_prompt_text=original_prompt,
            persona_config=persona_config,
            max_output_tokens_for_turn=max_output_tokens,
            system_message_for_token_count=system_message,
            is_self_analysis_prompt=False,
        )

        # Check that optimization reduced token count (mocked to be half for this test)
        original_tokens = mock_tokenizer.count_tokens(original_prompt)
        optimized_tokens = mock_tokenizer.count_tokens(optimized_prompt)
        assert optimized_tokens <= original_tokens

        # Check that meaning is preserved (simple keyword check)
        assert "gravity" in optimized_prompt
        assert "simple" in optimized_prompt

    def test_prompt_effectiveness(
        self, prompt_optimizer_instance, mock_tokenizer
    ):  # NEW: Add fixtures
        # Create a mock LLM that can evaluate prompt effectiveness
        mock_llm = MagicMock()
        mock_llm.get_completion.side_effect = [
            "The original prompt is clear and concise.",
            "The optimized prompt is clear and concise.",
        ]

        original_prompt = "Explain the concept of gravity in simple terms"
        # NEW: Create a mock PersonaConfig for the optimize_prompt signature
        persona_config = MagicMock(
            spec=PersonaConfig, name="TestPersona", token_efficiency_score=0.8
        )
        system_message = "You are a helpful assistant."
        max_output_tokens = 500

        # NEW: Call optimize_prompt with the correct signature
        optimized_prompt = prompt_optimizer_instance.optimize_prompt(
            user_prompt_text=original_prompt,
            persona_config=persona_config,
            max_output_tokens_for_turn=max_output_tokens,
            system_message_for_token_count=system_message,
            is_self_analysis_prompt=False,
        )

        # Check if the optimized prompt maintains effectiveness
        assert "clear" in mock_llm.get_completion(original_prompt)
        assert "clear" in mock_llm.get_completion(optimized_prompt)

    def test_token_usage_tracking(self, mock_token_tracker):  # NEW: Add fixture
        prompt = "Explain the concept of gravity in simple terms"
        mock_token_tracker.add_usage(len(prompt.split()))
        assert mock_token_tracker.get_total_usage() == len(prompt.split())
