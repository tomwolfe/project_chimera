import pytest
from src.self_improvement_analyzer import SelfImprovementAnalyzer


# Mocking necessary components for SelfImprovementAnalyzer
@pytest.fixture
def mock_metrics_collector():
    return MagicMock()


@pytest.fixture
def mock_prompt_engineer():
    return MagicMock()


@pytest.fixture
def mock_llm_provider():
    return MagicMock()


def test_self_improvement_analyzer_initialization(
    mock_metrics_collector, mock_prompt_engineer, mock_llm_provider
):
    """Tests the initialization of the SelfImprovementAnalyzer."""
    analyzer = SelfImprovementAnalyzer(
        metrics_collector=mock_metrics_collector,
        prompt_engineer=mock_prompt_engineer,
        llm_provider=mock_llm_provider,
    )
    assert analyzer.metrics_collector == mock_metrics_collector
    assert analyzer.prompt_engineer == mock_prompt_engineer
    assert analyzer.llm_provider == mock_llm_provider


def test_analyze_self_improvement_plan_success(
    mock_metrics_collector, mock_prompt_engineer, mock_llm_provider
):
    """Tests the successful generation of a self-improvement plan."""
    # Mocking the LLM provider to return a structured plan
    mock_llm_provider.generate_content.return_value = '{"plan": "Improve prompt quality", "steps": ["Refine prompt templates", "Add few-shot examples"]}'
    # Mocking the output parser if it's used internally by analyze_self_improvement_plan
    # For simplicity, assuming generate_content returns a parsable string directly or parse_llm_output is called

    analyzer = SelfImprovementAnalyzer(
        metrics_collector=mock_metrics_collector,
        prompt_engineer=mock_prompt_engineer,
        llm_provider=mock_llm_provider,
    )

    # Mocking the prompt engineer to provide a specific prompt
    mock_prompt_engineer.generate_self_improvement_prompt.return_value = (
        "Analyze performance metrics and suggest improvements."
    )

    improvement_plan = analyzer.analyze_self_improvement_plan()

    assert isinstance(improvement_plan, dict)
    assert "plan" in improvement_plan
    assert "steps" in improvement_plan
    assert improvement_plan["plan"] == "Improve prompt quality"
    mock_llm_provider.generate_content.assert_called_once_with(
        "Analyze performance metrics and suggest improvements."
    )
    mock_prompt_engineer.generate_self_improvement_prompt.assert_called_once()


def test_analyze_self_improvement_plan_api_error(
    mock_metrics_collector, mock_prompt_engineer, mock_llm_provider
):
    """Tests handling of API errors during self-improvement plan generation."""
    mock_llm_provider.generate_content.side_effect = Exception("LLM API Error")
    mock_prompt_engineer.generate_self_improvement_prompt.return_value = (
        "Analyze performance metrics and suggest improvements."
    )

    analyzer = SelfImprovementAnalyzer(
        metrics_collector=mock_metrics_collector,
        prompt_engineer=mock_prompt_engineer,
        llm_provider=mock_llm_provider,
    )

    with pytest.raises(Exception, match="LLM API Error"):
        analyzer.analyze_self_improvement_plan()
    mock_llm_provider.generate_content.assert_called_once()
    mock_prompt_engineer.generate_self_improvement_prompt.assert_called_once()


def test_analyze_self_improvement_plan_parsing_error(
    mock_metrics_collector, mock_prompt_engineer, mock_llm_provider
):
    """Tests handling of errors when parsing the LLM output."""
    # Simulate an LLM response that cannot be parsed as JSON
    mock_llm_provider.generate_content.return_value = "Invalid JSON response"
    mock_prompt_engineer.generate_self_improvement_prompt.return_value = (
        "Analyze performance metrics and suggest improvements."
    )

    analyzer = SelfImprovementAnalyzer(
        metrics_collector=mock_metrics_collector,
        prompt_engineer=mock_prompt_engineer,
        llm_provider=mock_llm_provider,
    )

    with pytest.raises(ValueError, match="Could not parse LLM output as JSON"):
        analyzer.analyze_self_improvement_plan()
    mock_llm_provider.generate_content.assert_called_once()
    mock_prompt_engineer.generate_self_improvement_prompt.assert_called_once()


# Consider adding tests for specific metrics collection and prompt engineering interactions if those modules are complex.
