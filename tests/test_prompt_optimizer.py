import pytest
from unittest.mock import MagicMock
from src.utils.prompt_optimizer import PromptOptimizer
from src.llm_tokenizers.base import Tokenizer
import json
from src.config.settings import ChimeraSettings


@pytest.fixture
def mock_summarizer_pipeline():
    """Provides a mock Hugging Face summarization pipeline."""
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = [{"summary_text": "Mock summary."}]
    mock_pipeline.tokenizer.model_max_length = 1024  # Simulate distilbart's max input
    return mock_pipeline


@pytest.fixture
def mock_tokenizer():
    """Provides a mock Tokenizer instance."""
    tokenizer = MagicMock(spec=Tokenizer)
    # Simple token counting: 1 token per 4 characters, ensure at least 1 token
    tokenizer.count_tokens.side_effect = lambda text: max(1, len(text) // 4)

    def mock_truncate(text, max_tokens, truncation_indicator=""):
        # Simulate the actual tokenizer's behavior more closely
        # Calculate tokens for the indicator first
        indicator_tokens = tokenizer.count_tokens(truncation_indicator)
        # Determine how many tokens are left for the main text
        effective_max_tokens_for_text = max(1, max_tokens - indicator_tokens)

        current_tokens_of_text = tokenizer.count_tokens(text)
        if current_tokens_of_text <= effective_max_tokens_for_text:
            return text

        # Simulate truncation: cut characters to roughly match effective_max_tokens_for_text
        # This is a heuristic, but ensures the length changes and token count is respected.
        target_char_len = effective_max_tokens_for_text * 4

        # Ensure target_char_len doesn't exceed original text length if text is already short
        if len(text) <= target_char_len:
            return text  # Should not happen if current_tokens_of_text > effective_max_tokens_for_text

        return text[:target_char_len] + truncation_indicator

    tokenizer.truncate_to_token_limit.side_effect = mock_truncate
    return tokenizer


@pytest.fixture
def mock_settings():
    """Provides a mock ChimeraSettings instance."""
    settings = MagicMock(spec=ChimeraSettings)
    settings.default_max_input_tokens_per_persona = 4000
    settings.max_tokens_per_persona = {
        "HighTokenPersona": 1000,  # Specific low limit for testing truncation
        "Self_Improvement_Analyst": 4000,  # Specific limit for structured optimization
    }
    return settings


@pytest.fixture
def prompt_optimizer_instance(mock_tokenizer, mock_settings, mock_summarizer_pipeline):
    """Provides a PromptOptimizer instance with mocked dependencies."""
    return PromptOptimizer(
        tokenizer=mock_tokenizer,
        settings=mock_settings,
        summarizer_pipeline=mock_summarizer_pipeline,
    )


def test_optimize_prompt_within_limit(prompt_optimizer_instance):
    """Test that prompt is not optimized if already within limits."""
    prompt = "This is a short prompt."
    persona_name = "GeneralPersona"
    max_output_tokens_for_turn = 1000  # Irrelevant for input optimization here

    optimized_prompt = prompt_optimizer_instance.optimize_prompt(
        prompt, persona_name, max_output_tokens_for_turn
    )

    assert optimized_prompt == prompt
    prompt_optimizer_instance.tokenizer.count_tokens.assert_called_once_with(prompt)
    prompt_optimizer_instance.tokenizer.truncate_to_token_limit.assert_not_called()


def test_optimize_prompt_exceeds_default_limit(prompt_optimizer_instance):
    """Test that prompt is truncated if it exceeds the default persona input limit."""
    # Make long_prompt actually exceed the default limit (4000 tokens * 4 chars/token = 16000 chars)
    long_prompt = "A" * 20000  # This is 5000 tokens, exceeding 4000 token limit
    persona_name = "GeneralPersona"
    max_output_tokens_for_turn = 1000

    optimized_prompt = prompt_optimizer_instance.optimize_prompt(
        long_prompt, persona_name, max_output_tokens_for_turn
    )

    expected_truncated_tokens = (
        prompt_optimizer_instance.settings.default_max_input_tokens_per_persona
    )  # 4000 tokens
    assert optimized_prompt != long_prompt  # Should be truncated
    assert len(optimized_prompt) < len(long_prompt)
    assert (
        prompt_optimizer_instance.tokenizer.count_tokens(optimized_prompt)
        <= expected_truncated_tokens
    )
    prompt_optimizer_instance.tokenizer.truncate_to_token_limit.assert_called_once()
    assert "[TRUNCATED - focusing on most critical aspects]" in optimized_prompt


def test_optimize_prompt_exceeds_specific_persona_limit(prompt_optimizer_instance):
    """Test that prompt is truncated if it exceeds a persona-specific input limit."""
    # Make long_prompt actually exceed the specific persona limit (1000 tokens * 4 chars/token = 4000 chars)
    long_prompt = (
        "B" * 5000
    )  # This is 1250 tokens, exceeding 1000 token limit for HighTokenPersona
    persona_name = (
        "HighTokenPersona"  # Has a specific limit of 1000 tokens (4000 chars)
    )
    max_output_tokens_for_turn = 1000

    optimized_prompt = prompt_optimizer_instance.optimize_prompt(
        long_prompt, persona_name, max_output_tokens_for_turn
    )

    expected_truncated_tokens = (
        prompt_optimizer_instance.settings.max_tokens_per_persona[persona_name]
    )  # 1000 tokens
    assert optimized_prompt != long_prompt  # Should be truncated
    assert len(optimized_prompt) < len(long_prompt)
    assert (
        prompt_optimizer_instance.tokenizer.count_tokens(optimized_prompt)
        <= expected_truncated_tokens
    )
    prompt_optimizer_instance.tokenizer.truncate_to_token_limit.assert_called_once()
    assert "[TRUNCATED - focusing on most critical aspects]" in optimized_prompt


def test_optimize_prompt_self_improvement_structured_truncation(
    prompt_optimizer_instance, mock_settings
):
    """Test structured truncation for Self_Improvement_Analyst persona."""
    # This prompt structure mimics the Self_Improvement_Analyst prompt
    # Make the prompt long enough to exceed the persona's input limit (4000 tokens)
    long_self_improvement_prompt = (
        """
Initial Problem: This is a very long initial problem description that needs to be truncated.
It has many sentences and words to ensure it exceeds the token limit. """
        + ("X" * (4000 * 4))
        + """

Relevant Code Context:
This is a very long code context section.
It contains many lines of code and explanations.
This section should be truncated aggressively.

Debate History:
This is a very long debate history section.
It contains many turns of conversation.
This section should be truncated even more aggressively.

Objective Metrics and Analysis:
This is a very long metrics section.
It contains detailed metrics and analysis.
This section should also be truncated.

---
CRITICAL JSON OUTPUT INSTRUCTIONS: ABSOLUTELY MUST BE FOLLOWED
1. MUST BE A SINGLE, VALID JSON OBJECT. NO ARRAYS.
2. NO NUMBERED ARRAY ELEMENTS.
3. ABSOLUTELY NO CONVERSATIONAL TEXT, MARKDOWN FENCES, OR EXPLANATIONS OUTSIDE JSON.
4. STRICTLY ADHERE TO SCHEMA.
5. USE DOUBLE QUOTES.
6. ENSURE COMMAS. NO TRAILING COMMAS.
7. PROPER JSON ARRAY SYNTAX: [{"key": "value"}, {"key": "value"}].
8. Include `malformed_blocks` field (even if empty).
---
"""
    )
    persona_name = "Self_Improvement_Analyst"
    max_output_tokens_for_turn = 1000  # Assume some output tokens

    # Self_Improvement_Analyst has a limit of 4000 tokens (~16000 chars)
    # The prompt is now designed to exceed this.

    # Mock the tokenizer's count_tokens to make the prompt appear longer for the initial check
    original_count_tokens = prompt_optimizer_instance.tokenizer.count_tokens
    # The mock tokenizer's default side_effect is len(text) // 4.
    # The prompt is now > 16000 chars, so it will be > 4000 tokens.

    optimized_prompt = prompt_optimizer_instance.optimize_prompt(
        long_self_improvement_prompt, persona_name, max_output_tokens_for_turn
    )

    # Restore original mock
    prompt_optimizer_instance.tokenizer.count_tokens = original_count_tokens

    # Assert that the prompt was indeed optimized (truncated)
    assert len(optimized_prompt) < len(long_self_improvement_prompt)
    assert "[TRUNCATED - focusing on most critical aspects]" in optimized_prompt

    # Assert that critical sections are likely preserved more than less critical ones
    # This is a qualitative check, but we can look for keywords.
    assert "Initial Problem:" in optimized_prompt
    assert "CRITICAL JSON OUTPUT INSTRUCTIONS:" in optimized_prompt

    # Assert that less critical sections are likely truncated
    assert (
        "Relevant Code Context:" in optimized_prompt
    )  # Should still be there, but truncated
    assert "Debate History:" in optimized_prompt  # Should still be there, but truncated


def test_optimize_debate_history_within_limit(prompt_optimizer_instance):
    """Test that debate history is not optimized if already within limits."""
    history_str = json.dumps([{"persona": "A", "output": "Short"}])
    max_tokens = 100

    optimized_history = prompt_optimizer_instance.optimize_debate_history(
        history_str, max_tokens
    )

    assert optimized_history == history_str
    prompt_optimizer_instance.tokenizer.count_tokens.assert_called_once_with(
        history_str
    )
    prompt_optimizer_instance.tokenizer.truncate_to_token_limit.assert_not_called()


def test_optimize_debate_history_exceeds_limit(prompt_optimizer_instance):
    """Test that debate history is truncated if it exceeds the limit."""
    long_history_str = json.dumps(
        [{"persona": f"Persona{i}", "output": "Long output " * 50} for i in range(10)]
    )
    max_tokens = 50

    optimized_history = prompt_optimizer_instance.optimize_debate_history(
        long_history_str, max_tokens
    )

    assert len(optimized_history) < len(long_history_str)
    assert (
        prompt_optimizer_instance.tokenizer.count_tokens(optimized_history)
        <= max_tokens
    )
    assert "... (debate history further summarized/truncated...)" in optimized_history
    prompt_optimizer_instance.tokenizer.truncate_to_token_limit.assert_called_once()
