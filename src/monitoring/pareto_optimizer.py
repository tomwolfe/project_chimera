"""
Pareto Optimizer for Project Chimera following the 80/20 principle.
Automatically identifies and suggests optimizations that provide 80% of the benefit with 20% of the effort.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

from src.monitoring.performance_logger import get_pareto_analyzer
from src.monitoring.system_monitor import get_system_monitor

logger = logging.getLogger(__name__)


class OptimizationPriority(Enum):
    """Priority levels for optimizations based on 80/20 impact."""
    CRITICAL = "critical"  # High impact, low effort
    HIGH = "high"          # High impact, medium effort
    MEDIUM = "medium"      # Medium impact
    LOW = "low"            # Low impact


@dataclass
class OptimizationRecommendation:
    """Represents a single optimization recommendation following 80/20 principle."""
    title: str
    component: str
    priority: OptimizationPriority
    impact_percentage: float  # Estimated impact (0-100%)
    effort_level: str  # "low", "medium", "high"
    estimated_savings: Dict[str, Any]  # e.g., {"tokens": 1000, "time": 2.5, "cost_usd": 0.0015}
    description: str
    implementation_steps: List[str]
    expected_timeline: str  # e.g., "immediate", "short", "medium", "long"
    confidence_level: float  # 0-1 confidence in the recommendation


class ParetoOptimizer:
    """
    Optimizer that applies the 80/20 principle to identify the highest-impact,
    lowest-effort improvements in Project Chimera.
    """

    def __init__(self):
        self.system_monitor = get_system_monitor()
        self.pareto_analyzer = get_pareto_analyzer()
        self.recommendations_cache = {}
        self.last_analysis_time = None

    def generate_optimizations(self, force_refresh: bool = False) -> List[OptimizationRecommendation]:
        """
        Generate optimization recommendations based on 80/20 analysis.

        Args:
            force_refresh: Whether to force a new analysis even if cache is recent

        Returns:
            List of optimization recommendations ordered by impact/effort ratio
        """
        current_time = datetime.now()

        # Cache for 5 minutes to avoid excessive computation
        if not force_refresh and self.last_analysis_time and \
           (current_time - self.last_analysis_time).seconds < 300:  # 5 minutes
            return self.recommendations_cache.get("recommendations", [])

        recommendations = []

        # Analyze current system performance
        analysis_result = self.system_monitor.get_summary_report()
        pareto_analysis = analysis_result.get("pareto_analysis", {})

        # Generate recommendations based on different analysis areas
        recommendations.extend(self._analyze_performance_optimizations(pareto_analysis))
        recommendations.extend(self._analyze_token_efficiency_optimizations(pareto_analysis))
        recommendations.extend(self._analyze_error_reduction_optimizations(pareto_analysis))
        recommendations.extend(self._analyze_resource_efficiency_optimizations(pareto_analysis))

        # Sort by impact/effort ratio (80/20 principle)
        recommendations.sort(key=lambda rec:
                           self._calculate_impact_effort_ratio(rec),
                           reverse=True)

        # Update cache
        self.recommendations_cache["recommendations"] = recommendations
        self.last_analysis_time = current_time

        return recommendations

    def _calculate_impact_effort_ratio(self, recommendation: OptimizationRecommendation) -> float:
        """Calculate impact/effort ratio for prioritization."""
        impact = recommendation.impact_percentage / 100.0
        effort = {"low": 1, "medium": 2, "high": 3}.get(recommendation.effort_level, 2)
        return impact / effort

    def _analyze_performance_optimizations(self, pareto_analysis: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Generate recommendations based on performance bottlenecks."""
        recommendations = []

        bottlenecks = pareto_analysis.get("performance_bottlenecks", [])

        for bottleneck in bottlenecks:
            persona = bottleneck.get("persona", "unknown")
            avg_duration = bottleneck.get("avg_duration", 0)
            success_rate = bottleneck.get("success_rate", 1.0)

            if avg_duration > 5.0:  # If taking more than 5 seconds on average
                impact_pct = min(80, avg_duration * 5)  # Higher duration = higher potential impact
                effort_level = "medium" if avg_duration > 10 else "low"

                recommendations.append(
                    OptimizationRecommendation(
                        title=f"Reduce {persona} processing time",
                        component=persona,
                        priority=OptimizationPriority.HIGH if avg_duration > 10 else OptimizationPriority.MEDIUM,
                        impact_percentage=impact_pct,
                        effort_level=effort_level,
                        estimated_savings={
                            "time": avg_duration * 0.3,  # Conservative 30% reduction
                            "avg_duration_saved": avg_duration * 0.3
                        },
                        description=f"{persona} is a performance bottleneck with average duration of {avg_duration:.2f}s.",
                        implementation_steps=[
                            f"Optimize {persona} prompt complexity",
                            f"Reduce {persona} response length requirements",
                            f"Add timeout constraints for {persona} responses"
                        ],
                        expected_timeline="short",
                        confidence_level=0.7
                    )
                )

        return recommendations

    def _analyze_token_efficiency_optimizations(self, pareto_analysis: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Generate recommendations based on token usage inefficiencies."""
        recommendations = []

        token_analysis = pareto_analysis.get("token_efficiency", {})
        inefficient_personas = token_analysis.get("most_inefficient_personas", [])

        for persona_info in inefficient_personas:
            persona = persona_info.get("persona", "unknown")
            avg_tokens_per_call = persona_info.get("avg_tokens_per_call", 0)
            total_tokens = persona_info.get("total_tokens", 0)

            if avg_tokens_per_call > 4000:  # Threshold for high token usage
                potential_savings = avg_tokens_per_call * 0.25  # Conservative 25% reduction
                impact_pct = min(60, avg_tokens_per_call / 100)  # Higher usage = higher impact

                recommendations.append(
                    OptimizationRecommendation(
                        title=f"Optimize {persona} token usage",
                        component=persona,
                        priority=OptimizationPriority.HIGH if avg_tokens_per_call > 8000 else OptimizationPriority.MEDIUM,
                        impact_percentage=impact_pct,
                        effort_level="low",
                        estimated_savings={
                            "tokens": potential_savings,
                            "cost_usd": potential_savings * 0.000015,  # Google's pricing
                            "cost_reduction_percentage": 25
                        },
                        description=f"{persona} consumes excessive tokens with average {avg_tokens_per_call:.0f} per call.",
                        implementation_steps=[
                            f"Implement stricter token limits for {persona}",
                            f"Optimize {persona} prompt to be more concise",
                            f"Add token budget warnings for {persona}"
                        ],
                        expected_timeline="immediate",
                        confidence_level=0.8
                    )
                )

        return recommendations

    def _analyze_error_reduction_optimizations(self, pareto_analysis: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Generate recommendations based on error patterns."""
        recommendations = []

        error_analysis = pareto_analysis.get("error_patterns", {})
        top_errors = error_analysis.get("top_errors", [])
        error_impact = error_analysis.get("error_impact_analysis", {})

        total_errors = error_impact.get("total_errors", 0)

        for error_info in top_errors:
            error_type = error_info.get("error_type", "unknown")
            count = error_info.get("count", 0)
            percentage = error_info.get("percentage", 0)

            if total_errors > 0 and percentage > 5:  # Errors > 5% of total
                impact_pct = min(90, percentage * 2)  # Higher percentage = higher impact

                recommendations.append(
                    OptimizationRecommendation(
                        title=f"Reduce {error_type} occurrences",
                        component=error_type,
                        priority=OptimizationPriority.HIGH if percentage > 20 else OptimizationPriority.MEDIUM,
                        impact_percentage=impact_pct,
                        effort_level="medium" if error_type == "LLMProviderError" else "low",
                        estimated_savings={
                            "errors_reduced": count * 0.5,  # Conservative 50% reduction
                            "success_rate_improvement": percentage * 0.5
                        },
                        description=f"{error_type} represents {percentage:.1f}% of all errors.",
                        implementation_steps=[
                            f"Add specific error handling for {error_type}",
                            f"Implement retry logic with exponential backoff for {error_type}",
                            f"Add validation to prevent {error_type} before occurrence"
                        ],
                        expected_timeline="short",
                        confidence_level=0.6
                    )
                )

        return recommendations

    def _analyze_resource_efficiency_optimizations(self, pareto_analysis: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Generate recommendations based on resource usage."""
        recommendations = []

        resource_analysis = pareto_analysis.get("resource_usage", {})

        current_cpu = resource_analysis.get("current_cpu_percent", 0)
        current_memory = resource_analysis.get("current_memory_mb", 0)
        cpu_trend = resource_analysis.get("cpu_trend", "stable")
        memory_trend = resource_analysis.get("memory_trend", "stable")

        # CPU optimization recommendations
        if current_cpu > 70 or cpu_trend == "increasing":
            impact_pct = min(80, current_cpu)
            recommendations.append(
                OptimizationRecommendation(
                    title="Optimize CPU resource utilization",
                    component="system",
                    priority=OptimizationPriority.HIGH if current_cpu > 80 else OptimizationPriority.MEDIUM,
                    impact_percentage=impact_pct,
                    effort_level="medium",
                    estimated_savings={
                        "cpu_reduction_percentage": 15,
                        "system_performance_improvement": 10
                    },
                    description=f"CPU usage at {current_cpu:.1f}% with {cpu_trend} trend.",
                    implementation_steps=[
                        "Implement asynchronous processing for non-critical operations",
                        "Add process pooling and resource reuse",
                        "Optimize algorithm efficiency in high-CPU components"
                    ],
                    expected_timeline="medium",
                    confidence_level=0.7
                )
            )

        # Memory optimization recommendations
        if current_memory > 512 or memory_trend == "increasing":  # >512MB
            impact_pct = min(70, (current_memory / 1024) * 100)
            recommendations.append(
                OptimizationRecommendation(
                    title="Optimize memory resource utilization",
                    component="system",
                    priority=OptimizationPriority.HIGH if current_memory > 1024 else OptimizationPriority.MEDIUM,
                    impact_percentage=impact_pct,
                    effort_level="medium",
                    estimated_savings={
                        "memory_reduction_mb": current_memory * 0.2,
                        "memory_stability_improvement": 20
                    },
                    description=f"Memory usage at {current_memory:.1f}MB with {memory_trend} trend.",
                    implementation_steps=[
                        "Implement efficient caching strategies",
                        "Add memory cleanup and garbage collection",
                        "Optimize data structures to reduce memory footprint"
                    ],
                    expected_timeline="medium",
                    confidence_level=0.6
                )
            )

        return recommendations

    def apply_optimization(self, recommendation: OptimizationRecommendation) -> bool:
        """
        Apply a specific optimization (placeholder for more complex logic).

        Args:
            recommendation: The recommendation to apply

        Returns:
            True if successfully applied, False otherwise
        """
        # This would contain logic to actually apply the optimization
        # For now, just log the intention to apply
        logger.info(f"Optimization applied: {recommendation.title} - {recommendation.description}")

        # In a real implementation, this would modify system parameters,
        # update configuration, or trigger other optimization processes
        return True

    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate a comprehensive optimization report following 80/20 principle."""
        recommendations = self.generate_optimizations()

        # Calculate potential impact summary
        total_potential_time_savings = sum(
            rec.estimated_savings.get("time", 0) for rec in recommendations
        )

        total_potential_token_savings = sum(
            rec.estimated_savings.get("tokens", 0) for rec in recommendations
        )

        total_potential_cost_savings = sum(
            rec.estimated_savings.get("cost_usd", 0) for rec in recommendations
        )

        high_priority_count = sum(1 for rec in recommendations
                                 if rec.priority in [OptimizationPriority.CRITICAL, OptimizationPriority.HIGH])

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_recommendations": len(recommendations),
                "high_priority_recommendations": high_priority_count,
                "potential_time_savings_per_debate": f"{total_potential_time_savings:.2f}s",
                "potential_token_savings": f"{total_potential_token_savings:,.0f}",
                "potential_cost_savings_usd": f"${total_potential_cost_savings:.4f}"
            },
            "recommendations": [
                {
                    "title": rec.title,
                    "component": rec.component,
                    "priority": rec.priority.value,
                    "impact_percentage": rec.impact_percentage,
                    "effort_level": rec.effort_level,
                    "description": rec.description,
                    "expected_timeline": rec.expected_timeline,
                    "confidence_level": rec.confidence_level
                }
                for rec in recommendations[:10]  # Top 10 recommendations
            ],
            "pareto_principle_applied": True,
            "focus_area": "High impact, low effort optimizations"
        }

        return report


# Global optimizer instance
pareto_optimizer = ParetoOptimizer()


def get_pareto_optimizer() -> ParetoOptimizer:
    """Get the global Pareto optimizer instance."""
    return pareto_optimizer
