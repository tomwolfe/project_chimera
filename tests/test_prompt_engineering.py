import pytest
from src.utils.prompt_engineering import optimize_reasoning_prompt, create_reasoning_quality_metrics_prompt

class TestPromptEngineering:

    def test_optimize_reasoning_prompt_basic(self):
        """Test basic reasoning prompt optimization."""
        original_prompt = "Analyze the codebase for improvements."
        optimized = optimize_reasoning_prompt(original_prompt)

        # Check that critical elements were added
        assert "80/20" in optimized or "Pareto" in optimized.lower()
        assert "reasoning quality" in optimized.lower() or "robustness" in optimized.lower()
        assert "concise" in optimized.lower() or "token" in optimized.lower()
        assert "JSON" in optimized or "schema" in optimized

    def test_optimize_reasoning_prompt_already_optimized(self):
        """Test that already optimized prompts are not redundantly modified."""
        original_prompt = """Analyze the codebase.
        CRITICAL: Apply the 80/20 Pareto principle.
        PRIORITIZE: reasoning quality, robustness.
        IMPORTANT: Be concise. Target <2000 tokens.
        FORMAT: Your response MUST follow the SelfImprovementAnalysisOutputV1 JSON schema."""
        optimized = optimize_reasoning_prompt(original_prompt)
        assert optimized == original_prompt # Should not add duplicates

    def test_create_reasoning_quality_metrics_prompt(self):
        """Test creation of specialized reasoning quality metrics prompt."""
        metrics = {
            "performance_efficiency": {
                "token_usage_stats": {
                    "total_tokens": 5000,
                    "persona_token_usage": {
                        "Self_Improvement_Analyst": 3912,
                        "Devils_Advocate": 2500
                    }
                }
            },
            "reasoning_quality": {
                "overall_score": 0.65
            }
        }

        prompt = create_reasoning_quality_metrics_prompt(metrics)

        # Verify key elements are present
        assert "3912" in prompt  # Token count for high-usage persona
        assert "Analyze Project Chimera's reasoning quality" in prompt
        assert "80/20 principle" in prompt
        assert "top 1-2 most impactful changes" in prompt