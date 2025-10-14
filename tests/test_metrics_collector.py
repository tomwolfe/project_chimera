from unittest.mock import Mock, patch

from src.context.context_analyzer import CodebaseScanner
from src.models import ConfigurationAnalysisOutput, DeploymentAnalysisOutput
from src.self_improvement.metrics_collector import FocusedMetricsCollector
from src.token_tracker import TokenUsageTracker


class TestFocusedMetricsCollector:
    def setup_method(self):
        """Set up common test fixtures."""
        self.initial_prompt = "Test prompt for analysis"
        self.debate_history = [
            {
                "persona": "Test_Persona",
                "output": {
                    "general_output": "This is a test output with counter arguments"
                },
            }
        ]
        self.intermediate_steps = {
            "Total_Tokens_Used": 1000,
            "Total_Estimated_Cost_USD": 0.05,
            "Debate_History": self.debate_history,
        }

        # Mock tokenizer, llm_provider, etc.
        self.tokenizer = Mock()
        self.llm_provider = Mock()
        self.llm_provider.calculate_usd_cost.return_value = 0.05
        self.persona_manager = Mock()
        self.content_validator = Mock()

        # Create a mock CodebaseScanner
        self.codebase_scanner = Mock(spec=CodebaseScanner)
        self.codebase_scanner.raw_file_contents = {"test.py": "print('hello world')"}

        # Create a mock TokenUsageTracker
        self.token_tracker = Mock(spec=TokenUsageTracker)
        self.token_tracker.get_granular_usage_summary.return_value = {
            "total_tokens": 1000,
            "total_prompt_tokens": 700,
            "total_completion_tokens": 300,
            "persona_breakdown": {
                "Test_Persona": {
                    "total": 500,
                    "prompt": 350,
                    "completion": 150,
                    "successful_turns": 1,
                }
            },
        }

    def test_initialization(self):
        """Test initialization of FocusedMetricsCollector."""
        collector = FocusedMetricsCollector(
            initial_prompt=self.initial_prompt,
            debate_history=self.debate_history,
            intermediate_steps=self.intermediate_steps,
            tokenizer=self.tokenizer,
            llm_provider=self.llm_provider,
            persona_manager=self.persona_manager,
            content_validator=self.content_validator,
            codebase_scanner=self.codebase_scanner,
            token_tracker=self.token_tracker,
        )

        assert collector.initial_prompt == self.initial_prompt
        assert collector.debate_history == self.debate_history
        assert collector.intermediate_steps == self.intermediate_steps
        assert collector.raw_file_contents == {"test.py": "print('hello world')"}
        assert collector.tokenizer == self.tokenizer
        assert collector.llm_provider == self.llm_provider
        assert collector.persona_manager == self.persona_manager
        assert collector.content_validator == self.content_validator
        assert collector.codebase_scanner == self.codebase_scanner
        assert collector.token_tracker == self.token_tracker

    def test_identify_critical_metric(self):
        """Test identification of critical metric."""
        collector = FocusedMetricsCollector(
            initial_prompt=self.initial_prompt,
            debate_history=self.debate_history,
            intermediate_steps=self.intermediate_steps,
            tokenizer=self.tokenizer,
            llm_provider=self.llm_provider,
            persona_manager=self.persona_manager,
            content_validator=self.content_validator,
            codebase_scanner=self.codebase_scanner,
            token_tracker=self.token_tracker,
        )

        # Test with token efficiency above threshold (should be critical)
        collected_metrics = {
            "token_efficiency": 3000,  # Above threshold of 2000
            "impact_potential": 50,  # Above threshold of 40
            "fix_confidence": 80,  # Above threshold of 70
        }
        collector._identify_critical_metric(collected_metrics)
        # token_efficiency = 3000 - 2000 = 1000 (deviation) - highest
        assert collector.critical_metric == "token_efficiency"

        # Test with impact potential below threshold (should be critical)
        collected_metrics = {
            "token_efficiency": 1500,  # Below threshold of 2000
            "impact_potential": 20,  # Below threshold of 40 (deviation: 40-20=20)
            "fix_confidence": 60,  # Below threshold of 70 (deviation: 70-60=10)
        }
        collector._identify_critical_metric(collected_metrics)
        # For impact_potential: 40-20=20 (deviation), for fix_confidence: 70-60=10
        # So impact_potential should be more critical
        assert collector.critical_metric == "impact_potential"

    def test_analyze_reasoning_quality(self):
        """Test reasoning quality analysis."""
        collector = FocusedMetricsCollector(
            initial_prompt=self.initial_prompt,
            debate_history=self.debate_history,
            intermediate_steps=self.intermediate_steps,
            tokenizer=self.tokenizer,
            llm_provider=self.llm_provider,
            persona_manager=self.persona_manager,
            content_validator=self.content_validator,
            codebase_scanner=self.codebase_scanner,
            token_tracker=self.token_tracker,
        )

        analysis_output = {"test": "output"}
        collector.analyze_reasoning_quality(analysis_output)

        # Verify that reasoning quality metrics were set
        assert "reasoning_quality" in collector.collected_metrics
        reason_metrics = collector.collected_metrics["reasoning_quality"]
        assert "argument_strength_score" in reason_metrics
        assert "80_20_adherence_score" in reason_metrics
        assert "reasoning_depth" in reason_metrics

    def test_collect_configuration_analysis_classmethod(self):
        """Test the configuration analysis class method."""
        # Test when there are no configuration files (should handle gracefully)
        with patch("pathlib.Path.exists", return_value=False):
            result = FocusedMetricsCollector._collect_configuration_analysis(
                "/fake/path"
            )
            assert isinstance(result, ConfigurationAnalysisOutput)
            assert len(result.malformed_blocks) >= 0  # Should not crash

    def test_collect_deployment_robustness_metrics_classmethod(self):
        """Test the deployment robustness metrics class method."""
        # Test when there is no Dockerfile or requirements files (should handle gracefully)
        with patch("pathlib.Path.exists", return_value=False):
            result = FocusedMetricsCollector._collect_deployment_robustness_metrics(
                "/fake/path"
            )
            assert isinstance(result, DeploymentAnalysisOutput)
            assert result.dockerfile_present is False
            assert result.prod_requirements_present is False

    def test_collect_codebase_access_metrics(self):
        """Test codebase access metrics collection."""
        collector = FocusedMetricsCollector(
            initial_prompt=self.initial_prompt,
            debate_history=self.debate_history,
            intermediate_steps=self.intermediate_steps,
            tokenizer=self.tokenizer,
            llm_provider=self.llm_provider,
            persona_manager=self.persona_manager,
            content_validator=self.content_validator,
            codebase_scanner=self.codebase_scanner,
            token_tracker=self.token_tracker,
        )

        # Test with available codebase
        metrics = collector.collect_codebase_access_metrics()
        assert metrics["codebase_available"] is True
        assert metrics["critical_files_available_count"] >= 0

        # Test with empty codebase
        collector.raw_file_contents = {}
        metrics = collector.collect_codebase_access_metrics()
        assert metrics["codebase_available"] is False
        assert metrics["critical_files_available_count"] == 0

    def test_collect_token_usage_stats(self):
        """Test token usage statistics collection."""
        collector = FocusedMetricsCollector(
            initial_prompt=self.initial_prompt,
            debate_history=self.debate_history,
            intermediate_steps=self.intermediate_steps,
            tokenizer=self.tokenizer,
            llm_provider=self.llm_provider,
            persona_manager=self.persona_manager,
            content_validator=self.content_validator,
            codebase_scanner=self.codebase_scanner,
            token_tracker=self.token_tracker,
        )

        token_stats = collector._collect_token_usage_stats()

        assert "total_tokens" in token_stats
        assert "total_prompt_tokens" in token_stats
        assert "total_completion_tokens" in token_stats
        assert "total_cost_usd" in token_stats
        assert "token_efficiency" in token_stats

    def test_analyze_debate_efficiency(self):
        """Test debate efficiency analysis."""
        intermediate_steps = {
            "Debate_History": [{"persona": "Test_Persona", "output": "test"}],
            "debate_Tokens_Used": 500,
            "malformed_blocks": [],
            "Test_Persona_Tokens_Used": 200,
        }

        collector = FocusedMetricsCollector(
            initial_prompt=self.initial_prompt,
            debate_history=self.debate_history,
            intermediate_steps=intermediate_steps,
            tokenizer=self.tokenizer,
            llm_provider=self.llm_provider,
            persona_manager=self.persona_manager,
            content_validator=self.content_validator,
            codebase_scanner=self.codebase_scanner,
            token_tracker=self.token_tracker,
        )

        efficiency = collector._analyze_debate_efficiency()

        assert "num_turns" in efficiency
        assert "average_turn_tokens" in efficiency
        assert efficiency["num_turns"] == 1  # From intermediate_steps Debate_History

    def test_identify_successful_patterns(self):
        """Test identification of successful patterns."""
        collector = FocusedMetricsCollector(
            initial_prompt=self.initial_prompt,
            debate_history=self.debate_history,
            intermediate_steps=self.intermediate_steps,
            tokenizer=self.tokenizer,
            llm_provider=self.llm_provider,
            persona_manager=self.persona_manager,
            content_validator=self.content_validator,
            codebase_scanner=self.codebase_scanner,
            token_tracker=self.token_tracker,
        )

        # Test with sample records
        records = [
            {
                "success": True,
                "prompt_analysis": {
                    "reasoning_quality_metrics": {
                        "indicators": {"structured_output_request": True}
                    }
                },
                "persona_sequence": ["Self_Improvement_Analyst"],
                "CODE_CHANGES_SUGGESTED": [{"test": "change"}],
            },
            {
                "success": False,
                "prompt_analysis": {
                    "reasoning_quality_metrics": {
                        "indicators": {"structured_output_request": False}
                    }
                },
                "persona_sequence": ["Other_Persona"],
                "CODE_CHANGES_SUGGESTED": [],
            },
        ]

        patterns = collector.identify_successful_patterns(records)
        # Should return a dictionary with pattern success rates
        assert isinstance(patterns, dict)
        for _pattern, rate in patterns.items():
            assert isinstance(rate, float)
            assert 0.0 <= rate <= 1.0

    def test_analyze_historical_effectiveness(self):
        """Test historical effectiveness analysis."""
        collector = FocusedMetricsCollector(
            initial_prompt=self.initial_prompt,
            debate_history=self.debate_history,
            intermediate_steps=self.intermediate_steps,
            tokenizer=self.tokenizer,
            llm_provider=self.llm_provider,
            persona_manager=self.persona_manager,
            content_validator=self.content_validator,
            codebase_scanner=self.codebase_scanner,
            token_tracker=self.token_tracker,
        )

        # Test when history file doesn't exist (should handle gracefully)
        result = collector.analyze_historical_effectiveness()

        expected_keys = [
            "total_attempts",
            "success_rate",
            "top_performing_areas",
            "common_failure_modes",
            "historical_total_suggestions_processed",
            "historical_successful_suggestions",
            "historical_schema_validation_failures",
            "successful_patterns",
        ]

        for key in expected_keys:
            assert key in result

    def test_record_self_improvement_suggestion_outcome(self):
        """Test recording self-improvement suggestion outcomes."""
        collector = FocusedMetricsCollector(
            initial_prompt=self.initial_prompt,
            debate_history=self.debate_history,
            intermediate_steps=self.intermediate_steps,
            tokenizer=self.tokenizer,
            llm_provider=self.llm_provider,
            persona_manager=self.persona_manager,
            content_validator=self.content_validator,
            codebase_scanner=self.codebase_scanner,
            token_tracker=self.token_tracker,
        )

        # Initially should be 0
        assert collector._current_run_total_suggestions_processed == 0
        assert collector._current_run_successful_suggestions == 0

        # Record a successful suggestion
        collector.record_self_improvement_suggestion_outcome(
            "Test_Persona", True, False
        )
        assert collector._current_run_total_suggestions_processed == 1
        assert collector._current_run_successful_suggestions == 1

        # Record an unsuccessful suggestion
        collector.record_self_improvement_suggestion_outcome(
            "Test_Persona", False, True
        )
        assert collector._current_run_total_suggestions_processed == 2
        assert collector._current_run_successful_suggestions == 1  # Should still be 1

    def test_get_historical_self_improvement_success_rate(self):
        """Test getting historical success rate."""
        collector = FocusedMetricsCollector(
            initial_prompt=self.initial_prompt,
            debate_history=self.debate_history,
            intermediate_steps=self.intermediate_steps,
            tokenizer=self.tokenizer,
            llm_provider=self.llm_provider,
            persona_manager=self.persona_manager,
            content_validator=self.content_validator,
            codebase_scanner=self.codebase_scanner,
            token_tracker=self.token_tracker,
        )

        # Initially should be 0.0
        rate = collector._get_historical_self_improvement_success_rate()
        assert rate == 0.0

        # Set up some historical data
        collector._historical_total_suggestions_processed = 10
        collector._historical_successful_suggestions = 7
        rate = collector._get_historical_self_improvement_success_rate()
        assert rate == 0.7

    def test_get_historical_schema_validation_failures(self):
        """Test getting historical schema validation failures."""
        collector = FocusedMetricsCollector(
            initial_prompt=self.initial_prompt,
            debate_history=self.debate_history,
            intermediate_steps=self.intermediate_steps,
            tokenizer=self.tokenizer,
            llm_provider=self.llm_provider,
            persona_manager=self.persona_manager,
            content_validator=self.content_validator,
            codebase_scanner=self.codebase_scanner,
            token_tracker=self.token_tracker,
        )

        # Initially should return the defaultdict
        failures = collector._get_historical_schema_validation_failures()
        assert isinstance(failures, dict)

    def test_collect_all_metrics(self):
        """Test the main collect_all_metrics method."""
        with patch(
            "pathlib.Path.exists", return_value=False
        ):  # No config files for this test
            collector = FocusedMetricsCollector(
                initial_prompt=self.initial_prompt,
                debate_history=self.debate_history,
                intermediate_steps=self.intermediate_steps,
                tokenizer=self.tokenizer,
                llm_provider=self.llm_provider,
                persona_manager=self.persona_manager,
                content_validator=self.content_validator,
                codebase_scanner=self.codebase_scanner,
                token_tracker=self.token_tracker,
            )

            result = collector.collect_all_metrics()

            # Should return a dictionary with the expected metric categories
            expected_categories = [
                "performance_efficiency",
                "debate_efficiency",
                "maintainability",
                "configuration_analysis",
                "deployment_robustness",
                "code_quality",
                "security",
                "codebase_access",
            ]

            for category in expected_categories:
                assert category in result
