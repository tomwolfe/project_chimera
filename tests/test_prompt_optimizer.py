# tests/test_prompt_optimizer.py
import pytest
from unittest.mock import MagicMock

from src.utils.prompt_optimizer import PromptOptimizer
from src.llm_tokenizers.base import Tokenizer
from src.config.settings import ChimeraSettings
from src.models import PersonaConfig  # NEW: Import PersonaConfig


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


def test_prompt_optimizer_initialization(
    prompt_optimizer_instance, mock_tokenizer, mock_settings, mock_summarizer_pipeline
):
    """Tests that PromptOptimizer initializes correctly."""
    assert prompt_optimizer_instance.tokenizer == mock_tokenizer
    assert prompt_optimizer_instance.settings == mock_settings
    assert prompt_optimizer_instance.summarizer_pipeline == mock_summarizer_pipeline


def test_optimize_prompt_no_truncation_needed(prompt_optimizer_instance):
    """Tests optimize_prompt when the prompt is within limits."""
    user_prompt = "This is a short prompt."
    # NEW: Pass a mock PersonaConfig object
    persona_config = MagicMock(
        spec=PersonaConfig, name="TestPersona", token_efficiency_score=0.8
    )
    max_output_tokens = 500
    system_message = "You are a helpful assistant."

    optimized_prompt = prompt_optimizer_instance.optimize_prompt(
        user_prompt, persona_config, max_output_tokens, system_message
    )

    assert optimized_prompt == user_prompt
    prompt_optimizer_instance.tokenizer.count_tokens.assert_called()
    prompt_optimizer_instance.tokenizer.truncate_to_token_limit.assert_not_called()


def test_optimize_prompt_with_truncation(prompt_optimizer_instance, mock_tokenizer):
    """Tests optimize_prompt when truncation is required."""
    user_prompt = (
        "This is a very long prompt that definitely needs to be truncated because it exceeds the persona's input token limit. We need to make sure the optimization logic correctly reduces its size."
        * 5
    )
    # NEW: Pass a mock PersonaConfig object
    persona_config = MagicMock(
        spec=PersonaConfig, name="TestPersona", token_efficiency_score=0.8
    )
    max_output_tokens = 500
    system_message = "You are a helpful assistant."

    # Mock tokenizer to report a high token count for the long prompt
    mock_tokenizer.count_tokens.side_effect = [
        len(system_message) // 4,  # System message tokens
        (len(system_message) + len(user_prompt)) // 4,  # Full input tokens
        (len(system_message) + len(user_prompt))
        // 4,  # Full input tokens (again for truncation check)
        (len(system_message) + 100) // 4,  # Truncated user prompt tokens
    ]

    optimized_prompt = prompt_optimizer_instance.optimize_prompt(
        user_prompt, persona_config, max_output_tokens, system_message
    )

    assert optimized_prompt != user_prompt
    mock_tokenizer.truncate_to_token_limit.assert_called_once()
    assert "... (user prompt truncated)" in optimized_prompt


def test_optimize_debate_history_no_truncation(prompt_optimizer_instance):
    """Tests optimize_debate_history when no truncation is needed."""
    debate_history_str = '{"turn1": "output1", "turn2": "output2"}'
    max_tokens = 100

    optimized_history = prompt_optimizer_instance.optimize_debate_history(
        debate_history_str, max_tokens
    )

    assert optimized_history == debate_history_str
    prompt_optimizer_instance.tokenizer.count_tokens.assert_called()
    prompt_optimizer_instance.summarizer_pipeline.assert_not_called()


def test_optimize_debate_history_with_summarization(
    prompt_optimizer_instance, mock_tokenizer, mock_summarizer_pipeline
):
    """Tests optimize_debate_history when summarization is needed."""
    long_debate_history_str = (
        '{"turn1": "very long output...", "turn2": "another very long output..."}' * 10
    )
    max_tokens = 50

    # Mock tokenizer to report a high token count for the long history
    mock_tokenizer.count_tokens.side_effect = [
        (len(long_debate_history_str) + 100) // 4,  # Initial count
        (len(long_debate_history_str) + 100) // 4,  # For summarizer pre-truncation
        (len("Mock summary.") + 100) // 4,  # For final truncation check
    ]

    optimized_history = prompt_optimizer_instance.optimize_debate_history(
        long_debate_history_str, max_tokens
    )

    mock_summarizer_pipeline.assert_called_once()
    assert optimized_history == "Mock summary."
    assert prompt_optimizer_instance.tokenizer.truncate_to_token_limit.call_count >= 1
