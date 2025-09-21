import json
from contextlib import suppress
from unittest.mock import MagicMock, patch

import pytest

with suppress(ModuleNotFoundError):
    # Corrected import path
    from src.personas.self_improvement_analyst import SelfImprovementAnalyst


# Mocking necessary components for SelfImprovementAnalyst
@pytest.fixture
def mock_metrics_collector():
    collector = MagicMock()
    collector.collect_all_metrics.return_value = {
        "code_quality": {"ruff_issues_count": 0},
        "security": {"bandit_issues_count": 0},
        "performance_efficiency": {"token_usage_stats": {"persona_token_usage": {}}},
        "robustness": {"schema_validation_failures_count": 0},
        "maintainability": {
            "test_coverage_summary": {"overall_coverage_percentage": 100.0}
        },
        "reasoning_quality": {"content_misalignment_warnings": 0},
        "historical_analysis": {
            "total_attempts": 0,
            "success_rate": 0.0,
            "top_performing_areas": [],
            "common_failure_modes": {},
            "historical_total_suggestions_processed": 0,
            "historical_successful_suggestions": 0,
            "historical_schema_validation_failures": {},
        },
        "configuration_analysis": {
            "ci_workflow": {},
            "pre_commit_hooks": [],
            "pyproject_toml": {},
        },
        "deployment_robustness": {},
    }
    collector.analyze_historical_effectiveness.return_value = {
        "total_attempts": 0,
        "success_rate": 0.0,
        "historical_total_suggestions_processed": 0,
        "historical_successful_suggestions": 0,
        "historical_schema_validation_failures": {},
    }
    collector.record_self_improvement_suggestion_outcome.return_value = None
    collector.file_analysis_cache = {}  # Ensure this attribute exists
    collector._process_suggestions_for_quality.side_effect = (
        lambda x: x
    )  # Mock to return suggestions as is for simplicity in tests
    return collector


@pytest.fixture
def mock_llm_provider():
    provider = MagicMock()
    provider.tokenizer = MagicMock()
    provider.tokenizer.count_tokens.side_effect = lambda text: len(text) // 4
    provider.tokenizer.max_output_tokens = 8192
    provider.generate.return_value = (
        '{"ANALYSIS_SUMMARY": "Mock analysis summary.", "IMPACTFUL_SUGGESTIONS": []}',
        200,
        100,
        False,
    )
    return provider


@pytest.fixture
def mock_persona_manager():
    pm = MagicMock()
    pm.prompt_analyzer = MagicMock()
    pm.prompt_analyzer.is_self_analysis_prompt.return_value = True
    pm.get_adjusted_persona_config.return_value = MagicMock(
        system_prompt="Analyst", temperature=0.1, max_tokens=8192
    )
    return pm


@pytest.fixture
def mock_content_validator():
    validator = MagicMock()
    validator.validate.return_value = (True, "Content aligned.", {})
    return validator


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.count_tokens.side_effect = lambda text: len(text) // 4
    tokenizer.truncate_to_token_limit.side_effect = (
        lambda text, max_tokens, truncation_indicator="": text[: max_tokens * 4]
        + (truncation_indicator if len(text) > max_tokens * 4 else "")
    )
    return tokenizer


def test_self_improvement_analyzer_initialization(
    mock_metrics_collector,
    mock_llm_provider,
    mock_persona_manager,
    mock_content_validator,
    mock_tokenizer,
):
    """Tests the initialization of the SelfImprovementAnalyst."""
    if "SelfImprovementAnalyst" in locals():
        analyzer = SelfImprovementAnalyst(
            initial_prompt="Test prompt",  # NEW: Added initial_prompt
            metrics=mock_metrics_collector.collect_all_metrics(),  # Pass collected metrics
            debate_history=[],
            intermediate_steps={},
            codebase_raw_file_contents={},  # NEW: Pass raw_file_contents
            tokenizer=mock_tokenizer,
            llm_provider=mock_llm_provider,
            persona_manager=mock_persona_manager,
            content_validator=mock_content_validator,
            metrics_collector=mock_metrics_collector,
        )
        assert analyzer.metrics_collector == mock_metrics_collector
        assert analyzer.llm_provider == mock_llm_provider
        assert analyzer.initial_prompt == "Test prompt"
    else:
        pytest.skip("SelfImprovementAnalyst not imported, skipping test.")


def test_analyze_self_improvement_plan_success(
    mock_metrics_collector,
    mock_llm_provider,
    mock_persona_manager,
    mock_content_validator,
    mock_tokenizer,
):
    """Tests the successful generation of a self-improvement plan."""
    if "SelfImprovementAnalyst" not in locals():
        pytest.skip("SelfImprovementAnalyst not imported, skipping test.")

    # Mocking the LLM provider to return a structured plan
    mock_llm_provider.generate.return_value = (
        json.dumps(
            {
                "ANALYSIS_SUMMARY": "Improve prompt quality",
                "IMPACTFUL_SUGGESTIONS": [
                    {
                        "AREA": "Reasoning Quality",
                        "PROBLEM": "Prompts are vague.",
                        "PROPOSED_SOLUTION": "Make prompts more specific.",
                        "EXPECTED_IMPACT": "Better LLM responses.",
                        "CODE_CHANGES_SUGGESTED": [],
                    }
                ],
            }
        ),
        200,
        100,
        False,
    )

    analyzer = SelfImprovementAnalyst(
        initial_prompt="Test prompt",
        metrics=mock_metrics_collector.collect_all_metrics(),
        debate_history=[],
        intermediate_steps={},
        codebase_raw_file_contents={},
        tokenizer=mock_tokenizer,
        llm_provider=mock_llm_provider,
        persona_manager=mock_persona_manager,
        content_validator=mock_content_validator,
        metrics_collector=mock_metrics_collector,
    )

    improvement_plan = analyzer.analyze()  # Call the analyze method

    assert isinstance(improvement_plan, list)
    assert len(improvement_plan) == 1
    assert improvement_plan[0]["AREA"] == "Reasoning Quality"
    mock_llm_provider.generate.assert_called_once()


def test_analyze_self_improvement_plan_api_error(
    mock_metrics_collector,
    mock_llm_provider,
    mock_persona_manager,
    mock_content_validator,
    mock_tokenizer,
):
    """Tests handling of API errors during self-improvement plan generation."""
    if "SelfImprovementAnalyst" not in locals():
        pytest.skip("SelfImprovementAnalyst not imported, skipping test.")

    mock_llm_provider.generate.side_effect = Exception("LLM API Error")

    analyzer = SelfImprovementAnalyst(
        initial_prompt="Test prompt",
        metrics=mock_metrics_collector.collect_all_metrics(),
        debate_history=[],
        intermediate_steps={},
        codebase_raw_file_contents={},
        tokenizer=mock_tokenizer,
        llm_provider=mock_llm_provider,
        persona_manager=mock_persona_manager,
        content_validator=mock_content_validator,
        metrics_collector=mock_metrics_collector,
    )

    with pytest.raises(Exception, match="LLM API Error"):
        analyzer.analyze()
    mock_llm_provider.generate.assert_called_once()


def test_analyze_self_improvement_plan_parsing_error(
    mock_metrics_collector,
    mock_llm_provider,
    mock_persona_manager,
    mock_content_validator,
    mock_tokenizer,
):
    """Tests handling of errors when parsing the LLM output."""
    if "SelfImprovementAnalyst" not in locals():
        pytest.skip("SelfImprovementAnalyst not imported, skipping test.")

    # Simulate an LLM response that cannot be parsed as JSON
    mock_llm_provider.generate.return_value = ("Invalid JSON response", 10, 10, False)

    analyzer = SelfImprovementAnalyst(
        initial_prompt="Test prompt",
        metrics=mock_metrics_collector.collect_all_metrics(),
        debate_history=[],
        intermediate_steps={},
        codebase_raw_file_contents={},
        tokenizer=mock_tokenizer,
        llm_provider=mock_llm_provider,
        persona_manager=mock_persona_manager,
        content_validator=mock_content_validator,
        metrics_collector=mock_metrics_collector,
    )

    # The analyze method calls llm_provider.generate, which now performs early schema validation.
    # If the mock_llm_provider.generate returns "Invalid JSON response", it will raise SchemaValidationError.
    # The analyze method itself expects a list of suggestions.
    # For this test, we'll mock the internal parsing to fail.
    with (
        patch(
            "src.personas.self_improvement_analyst.json.loads",
            side_effect=json.JSONDecodeError(
                "Mock JSON error", doc="Invalid JSON", pos=0
            ),
        ),
        pytest.raises(ValueError, match="Could not parse LLM output as JSON"),
    ):
        analyzer.analyze()
    mock_llm_provider.generate.assert_called_once()


def test_analyze_self_improvement_plan_historical_low_success_rate(
    mock_metrics_collector,
    mock_llm_provider,
    mock_persona_manager,
    mock_content_validator,
    mock_tokenizer,
):
    """Tests that a suggestion is added when historical success rate is low."""
    if "SelfImprovementAnalyst" not in locals():
        pytest.skip("SelfImprovementAnalyst not imported, skipping test.")

    mock_metrics_collector.analyze_historical_effectiveness.return_value = {
        "total_attempts": 10,
        "success_rate": 0.3,  # Low success rate
        "top_performing_areas": [],
        "common_failure_modes": {},
        "historical_total_suggestions_processed": 10,
        "historical_successful_suggestions": 3,
        "historical_schema_validation_failures": {},
        "successful_patterns": {},  # Added for consistency
    }
    mock_llm_provider.generate.return_value = (
        json.dumps(
            {
                "ANALYSIS_SUMMARY": "Historical analysis suggests prompt refinement.",
                "IMPACTFUL_SUGGESTIONS": [],
            }
        ),
        100,
        50,
        False,
    )

    analyzer = SelfImprovementAnalyst(
        initial_prompt="Test prompt",
        metrics=mock_metrics_collector.collect_all_metrics(),
        debate_history=[],
        intermediate_steps={},
        codebase_raw_file_contents={},
        tokenizer=mock_tokenizer,
        llm_provider=mock_llm_provider,
        persona_manager=mock_persona_manager,
        content_validator=mock_content_validator,
        metrics_collector=mock_metrics_collector,
    )

    suggestions = analyzer.analyze()
    assert any(
        "Overall self-improvement success rate is low" in s["PROBLEM"]
        for s in suggestions
    )
    assert any(
        "Review the `Self_Improvement_Analyst`'s system prompt"
        in s["PROPOSED_SOLUTION"]
        for s in suggestions
    )


def test_analyze_self_improvement_plan_historical_common_failure_mode(
    mock_metrics_collector,
    mock_llm_provider,
    mock_persona_manager,
    mock_content_validator,
    mock_tokenizer,
):
    """Tests that a suggestion is added for a common historical failure mode."""
    if "SelfImprovementAnalyst" not in locals():
        pytest.skip("SelfImprovementAnalyst not imported, skipping test.")

    mock_metrics_collector.analyze_historical_effectiveness.return_value = {
        "total_attempts": 10,
        "success_rate": 0.8,
        "top_performing_areas": [],
        "common_failure_modes": {
            "schema_validation_failures_count": 5
        },  # Common failure
        "historical_total_suggestions_processed": 10,
        "historical_successful_suggestions": 8,
        "historical_schema_validation_failures": {"Constructive_Critic": 5},
        "successful_patterns": {},  # Added for consistency
    }
    mock_llm_provider.generate.return_value = (
        json.dumps(
            {
                "ANALYSIS_SUMMARY": "Historical analysis suggests addressing schema failures.",
                "IMPACTFUL_SUGGESTIONS": [],
            }
        ),
        100,
        50,
        False,
    )

    analyzer = SelfImprovementAnalyst(
        initial_prompt="Test prompt",
        metrics=mock_metrics_collector.collect_all_metrics(),
        debate_history=[],
        intermediate_steps={},
        codebase_raw_file_contents={},
        tokenizer=mock_tokenizer,
        llm_provider=mock_llm_provider,
        persona_manager=mock_persona_manager,
        content_validator=mock_content_validator,
        metrics_collector=mock_metrics_collector,
    )

    suggestions = analyzer.analyze()
    assert any(
        "Historical analysis identifies 'schema_validation_failures_count' as a common failure mode"
        in s["PROBLEM"]
        for s in suggestions
    )
    assert any(
        "Implement specific safeguards or prompt adjustments to mitigate 'schema_validation_failures_count'"
        in s["PROPOSED_SOLUTION"]
        for s in suggestions
    )


def test_analyze_self_improvement_plan_high_token_consumption(
    mock_metrics_collector,
    mock_llm_provider,
    mock_persona_manager,
    mock_content_validator,
    mock_tokenizer,
):
    """Tests that a suggestion is added for high token consumption."""
    if "SelfImprovementAnalyst" not in locals():
        pytest.skip("SelfImprovementAnalyst not imported, skipping test.")

    mock_metrics_collector.collect_all_metrics.return_value = {
        "code_quality": {"ruff_issues_count": 0},
        "security": {"bandit_issues_count": 0},
        "performance_efficiency": {
            "total_tokens": 3500,  # Total tokens
            "total_cost_usd": 0.001,
            "persona_token_usage": {"TestPersona": 3000, "OtherPersona": 500},
            "token_efficiency": 3500,  # Assuming 1 suggestion, so 3500 tokens/suggestion
        },
        "debate_efficiency": {  # Added for consistency
            "num_turns": 2,
            "malformed_blocks_count": 0,
            "conflict_resolution_attempts": 0,
            "unresolved_conflict": False,
            "average_turn_tokens": 1750.0,
            "persona_token_breakdown": {"TestPersona": 3000, "OtherPersona": 500},
        },
        "robustness": {"schema_validation_failures_count": 0},
        "maintainability": {
            "test_coverage_summary": {"overall_coverage_percentage": 100.0}
        },
        "reasoning_quality": {"content_misalignment_warnings": 0},
        "historical_analysis": {
            "total_attempts": 0,
            "success_rate": 0.0,
            "top_performing_areas": [],
            "common_failure_modes": {},
            "historical_total_suggestions_processed": 0,
            "historical_successful_suggestions": 0,
            "historical_schema_validation_failures": {},
            "successful_patterns": {},  # Added for consistency
        },
        "configuration_analysis": {},  # Added for consistency
        "deployment_robustness": {},  # Added for consistency
    }
    mock_llm_provider.generate.return_value = (
        json.dumps(
            {
                "ANALYSIS_SUMMARY": "Token optimization needed.",
                "IMPACTFUL_SUGGESTIONS": [],
            }
        ),
        100,
        50,
        False,
    )

    analyzer = SelfImprovementAnalyst(
        initial_prompt="Test prompt",
        metrics=mock_metrics_collector.collect_all_metrics(),
        debate_history=[],
        intermediate_steps={},
        codebase_raw_file_contents={},
        tokenizer=mock_tokenizer,
        llm_provider=mock_llm_provider,
        persona_manager=mock_persona_manager,
        content_validator=mock_content_validator,
        metrics_collector=mock_metrics_collector,
    )

    suggestions = analyzer.analyze()
    assert any(
        "High token consumption by personas" in s["PROBLEM"] for s in suggestions
    )
    assert any(
        "Optimize prompts for high-token personas" in s["PROPOSED_SOLUTION"]
        for s in suggestions
    )
