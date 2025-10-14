from src.token_tracker import TokenUsageTracker


class TestTokenUsageTracker:
    def test_token_usage_tracker_initialization(self):
        """Test that the token tracker initializes with correct default values."""
        tracker = TokenUsageTracker()
        assert tracker.budget == 128000
        assert tracker.current_usage == 0
        assert len(tracker.usage_history) == 0
        assert tracker.total_prompt_tokens == 0
        assert tracker.total_completion_tokens == 0
        assert len(tracker.persona_token_map) == 0
        assert len(tracker.persona_granular_map) == 0
        assert tracker.high_value_tokens == 0
        assert tracker.low_value_tokens == 0
        assert tracker._current_stage is None

    def test_token_usage_tracker_initialization_with_custom_budget(self):
        """Test that the token tracker initializes with a custom budget."""
        tracker = TokenUsageTracker(budget=50000)
        assert tracker.budget == 50000

    def test_token_usage_tracker_initialization_with_custom_max_history_items(self):
        """Test that the token tracker initializes with a custom max_history_items."""
        tracker = TokenUsageTracker(max_history_items=50)
        assert tracker.max_history_items == 50

    def test_record_usage_basic(self):
        """Test that tokens are recorded correctly."""
        tracker = TokenUsageTracker()
        tracker.record_usage(prompt_tokens=100, completion_tokens=50)

        assert tracker.current_usage == 150
        assert tracker.total_prompt_tokens == 100
        assert tracker.total_completion_tokens == 50
        assert len(tracker.usage_history) == 1

        # Check that history entry contains a timestamp and token count
        timestamp, tokens = tracker.usage_history[0]
        assert isinstance(timestamp, float)
        assert tokens == 150

    def test_record_usage_with_persona(self):
        """Test that tokens are attributed to a persona correctly."""
        tracker = TokenUsageTracker()
        tracker.record_usage(
            prompt_tokens=100, completion_tokens=50, persona="test_persona"
        )

        assert tracker.current_usage == 150
        assert tracker.total_prompt_tokens == 100
        assert tracker.total_completion_tokens == 50
        assert tracker.persona_token_map["test_persona"] == 150
        assert tracker.persona_granular_map["test_persona"]["total"] == 150
        assert tracker.persona_granular_map["test_persona"]["prompt"] == 100
        assert tracker.persona_granular_map["test_persona"]["completion"] == 50
        assert tracker.persona_granular_map["test_persona"]["successful_turns"] == 1

    def test_record_usage_with_unsuccessful_turn(self):
        """Test that tokens are recorded but successful turns not incremented for unsuccessful turns."""
        tracker = TokenUsageTracker()
        tracker.record_usage(
            prompt_tokens=100,
            completion_tokens=50,
            persona="test_persona",
            is_successful_turn=False,
        )

        assert tracker.persona_granular_map["test_persona"]["successful_turns"] == 0

    def test_record_usage_with_semantic_weighting(self):
        """Test that tokens are categorized based on stage."""
        tracker = TokenUsageTracker()

        # Set stage to final_synthesis and record tokens
        tracker.set_current_stage("final_synthesis")
        tracker.record_usage(prompt_tokens=100, completion_tokens=50)
        assert tracker.high_value_tokens == 150
        assert tracker.low_value_tokens == 0

        # Set stage to debate and record tokens
        tracker.set_current_stage("debate")
        tracker.record_usage(prompt_tokens=50, completion_tokens=25)
        assert tracker.high_value_tokens == 150
        assert tracker.low_value_tokens == 75

        # Set stage to context and record tokens
        tracker.set_current_stage("context")
        tracker.record_usage(prompt_tokens=30, completion_tokens=10)
        assert tracker.high_value_tokens == 150
        assert tracker.low_value_tokens == 115

    def test_get_consumption_rate(self):
        """Test that consumption rate is calculated correctly."""
        tracker = TokenUsageTracker(budget=1000)
        tracker.record_usage(prompt_tokens=250, completion_tokens=250)  # 500 total

        rate = tracker.get_consumption_rate()
        assert rate == 0.5  # 50%

    def test_get_consumption_rate_zero_budget(self):
        """Test that consumption rate returns 0.0 when budget is 0."""
        tracker = TokenUsageTracker(budget=0)
        tracker.record_usage(prompt_tokens=100, completion_tokens=50)

        rate = tracker.get_consumption_rate()
        assert rate == 0.0

    def test_get_consumption_rate_empty_history(self):
        """Test that consumption rate returns 0.0 when history is empty."""
        tracker = TokenUsageTracker()
        rate = tracker.get_consumption_rate()
        assert rate == 0.0

    def test_get_high_consumption_personas(self):
        """Test that high consumption personas are identified correctly."""
        tracker = TokenUsageTracker()
        tracker.record_usage(
            prompt_tokens=100, completion_tokens=50, persona="heavy_persona"
        )
        tracker.record_usage(
            prompt_tokens=10, completion_tokens=5, persona="light_persona"
        )

        # With default threshold (0.15), heavy_persona should be considered high consumption
        # (150 out of 165 total = 90.9% > 15%)
        high_consumers = tracker.get_high_consumption_personas()
        assert "heavy_persona" in high_consumers
        assert high_consumers["heavy_persona"] == 150

    def test_get_high_consumption_personas_no_high_consumers(self):
        """Test that empty dict is returned when no personas exceed threshold."""
        tracker = TokenUsageTracker()
        tracker.record_usage(prompt_tokens=50, completion_tokens=25, persona="persona1")
        tracker.record_usage(prompt_tokens=50, completion_tokens=25, persona="persona2")

        # Both personas have 50% each, so neither should exceed 75% threshold
        high_consumers = tracker.get_high_consumption_personas(threshold=0.75)
        assert high_consumers == {}

    def test_get_high_consumption_personas_zero_total(self):
        """Test that empty dict is returned when total is 0."""
        tracker = TokenUsageTracker()
        high_consumers = tracker.get_high_consumption_personas()
        assert high_consumers == {}

    def test_reset(self):
        """Test that the tracker resets to initial state."""
        tracker = TokenUsageTracker(budget=50000)
        tracker.set_current_stage("final_synthesis")
        tracker.record_usage(
            prompt_tokens=100, completion_tokens=50, persona="test_persona"
        )

        # Verify state before reset
        assert tracker.current_usage == 150
        assert tracker.total_prompt_tokens == 100
        assert tracker.total_completion_tokens == 50
        assert "test_persona" in tracker.persona_token_map
        assert tracker._current_stage == "final_synthesis"

        # Record with semantic weighting to increase high_value_tokens
        tracker.record_usage(
            prompt_tokens=50, completion_tokens=25
        )  # Both records should be high_value since stage is "final_synthesis"

        # Both records (100+50=150 and 50+25=75) should be counted as high_value since stage was "final_synthesis" when they were recorded
        assert tracker.high_value_tokens == 225  # 150 + 75

        # Perform reset
        tracker.reset()

        # Verify state after reset
        assert tracker.budget == 50000  # Budget should remain unchanged
        assert tracker.current_usage == 0
        assert tracker.total_prompt_tokens == 0
        assert tracker.total_completion_tokens == 0
        assert len(tracker.persona_token_map) == 0
        assert len(tracker.persona_granular_map) == 0
        assert tracker.high_value_tokens == 0
        assert tracker.low_value_tokens == 0
        assert tracker._current_stage is None

    def test_get_granular_usage_summary(self):
        """Test that granular usage summary returns correct data."""
        tracker = TokenUsageTracker()
        tracker.set_current_stage("final_synthesis")
        tracker.record_usage(
            prompt_tokens=100, completion_tokens=50, persona="test_persona"
        )

        summary = tracker.get_granular_usage_summary()

        assert summary["total_tokens"] == 150
        assert summary["total_prompt_tokens"] == 100
        assert summary["total_completion_tokens"] == 50
        assert "test_persona" in summary["persona_breakdown"]
        assert summary["semantic_breakdown"]["high_value_tokens"] == 150
        assert summary["semantic_breakdown"]["low_value_tokens"] == 0
        assert summary["current_stage"] == "final_synthesis"

    def test_history_truncation(self):
        """Test that usage history is truncated when exceeding max_history_items."""
        tracker = TokenUsageTracker(max_history_items=3)

        # Add more items than max_history_items
        for _i in range(5):
            tracker.record_usage(prompt_tokens=10, completion_tokens=5)

        # History should be truncated to last 3 items
        assert len(tracker.usage_history) == 3

        # Check that the last 3 entries are kept
        for entry in tracker.usage_history:
            timestamp, tokens = entry
            assert (
                tokens == 15
            )  # Each entry should have 15 tokens (10 prompt + 5 completion)
