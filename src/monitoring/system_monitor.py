"""
System monitoring module for Project Chimera following the 80/20 Pareto principle.
Tracks the most impactful metrics for system performance and identifies optimization opportunities.
"""

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import psutil

if TYPE_CHECKING:
    from src.token_tracker import TokenUsageTracker

# Constants for the monitoring system
MIN_MEMORY_METRICS_FOR_TREND = 2
MIN_METRICS_FOR_AVERAGE = 20


class MetricType(Enum):
    """Types of metrics collected by the monitoring system."""

    PERFORMANCE = "performance"
    TOKEN_USAGE = "token_usage"
    RESOURCE_UTILIZATION = "resource_utilization"
    ERROR_RATE = "error_rate"
    SYSTEM_HEALTH = "system_health"


@dataclass
class SystemMetric:
    """Represents a single system metric with timestamp and context."""

    metric_type: MetricType
    metric_name: str
    value: float
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


class SystemMonitor:
    """Central monitoring system that follows 80/20 principle by tracking most impactful metrics."""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history: deque[SystemMetric] = deque(maxlen=max_history)
        self.current_metrics: Dict[str, SystemMetric] = {}
        self.token_tracker: Optional[TokenUsageTracker] = None

        # Performance tracking
        self.debate_timings = []
        self.persona_performance = defaultdict(list)
        self.error_counts = defaultdict(int)

        # Resource monitoring
        self.process = psutil.Process()
        self.monitoring_active = False
        self.monitoring_thread = None

        # Callbacks for real-time monitoring
        self.callbacks = []

    def start_monitoring(self):
        """Start the system monitoring background thread."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(
                target=self._background_monitoring, daemon=True
            )
            self.monitoring_thread.start()

    def stop_monitoring(self):
        """Stop the system monitoring background thread."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)

    def _background_monitoring(self):
        """Background monitoring thread that periodically collects system metrics."""
        while self.monitoring_active:
            try:
                self._collect_resource_metrics()
                time.sleep(5)  # Collect every 5 seconds
            except Exception:
                # Don't let errors break the monitoring loop
                continue

    def _collect_resource_metrics(self):
        """Collect resource utilization metrics."""
        try:
            # CPU and Memory usage
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()

            # Add collected metrics
            self.record_metric(
                MetricType.RESOURCE_UTILIZATION,
                "cpu_percent",
                cpu_percent,
                tags=["system"],
                context={"process_id": self.process.pid},
            )

            self.record_metric(
                MetricType.RESOURCE_UTILIZATION,
                "memory_rss_mb",
                memory_info.rss / 1024 / 1024,  # Convert to MB
                tags=["system"],
                context={"process_id": self.process.pid},
            )

            self.record_metric(
                MetricType.RESOURCE_UTILIZATION,
                "memory_percent",
                memory_percent,
                tags=["system"],
                context={"process_id": self.process.pid},
            )
        except Exception:
            pass  # Don't break if resource monitoring fails

    def record_metric(
        self,
        metric_type: MetricType,
        metric_name: str,
        value: float,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ):
        """Record a single metric."""
        metric = SystemMetric(
            metric_type=metric_type,
            metric_name=metric_name,
            value=value,
            timestamp=datetime.now(),
            context=context or {},
            tags=tags or [],
        )

        # Store in current metrics dictionary
        self.current_metrics[f"{metric_type.value}.{metric_name}"] = metric
        self.metrics_history.append(metric)

        # Call registered callbacks
        from contextlib import suppress

        for callback in self.callbacks:
            with suppress(Exception):
                callback(metric)

    def record_debate_timing(self, duration: float, debate_id: str = None):
        """Record the duration of a complete debate."""
        self.debate_timings.append(duration)
        self.record_metric(
            MetricType.PERFORMANCE,
            "debate_duration_seconds",
            duration,
            context={"debate_id": debate_id} if debate_id else {},
        )

    def record_persona_performance(
        self,
        persona_name: str,
        success: bool,
        tokens_used: int,
        duration: float,
        debate_id: str = None,
    ):
        """Record performance metrics for a persona turn."""
        performance_record = {
            "success": success,
            "tokens_used": tokens_used,
            "duration": duration,
            "timestamp": datetime.now(),
            "debate_id": debate_id,
        }
        self.persona_performance[persona_name].append(performance_record)

        # Record metrics
        self.record_metric(
            MetricType.PERFORMANCE,
            f"persona_{persona_name}_duration",
            duration,
            context={"persona": persona_name, "debate_id": debate_id},
            tags=["persona"],
        )

        self.record_metric(
            MetricType.TOKEN_USAGE,
            f"persona_{persona_name}_tokens",
            tokens_used,
            context={"persona": persona_name, "debate_id": debate_id},
            tags=["persona"],
        )

    def record_error(
        self, error_type: str, error_message: str = None, debate_id: str = None
    ):
        """Record an error occurrence."""
        self.error_counts[error_type] += 1

        self.record_metric(
            MetricType.ERROR_RATE,
            f"error_{error_type}",
            1.0,
            context={
                "error_message": error_message,
                "debate_id": debate_id,
                "timestamp": datetime.now().isoformat(),
            },
        )

    def register_callback(self, callback):
        """Register a callback to be called whenever a metric is recorded."""
        self.callbacks.append(callback)

    def get_pareto_analysis(self) -> Dict[str, Any]:
        """Perform 80/20 analysis of system metrics."""
        analysis = {
            "performance_bottlenecks": self._analyze_performance_bottlenecks(),
            "token_efficiency": self._analyze_token_efficiency(),
            "error_patterns": self._analyze_error_patterns(),
            "resource_usage": self._analyze_resource_usage(),
        }
        return analysis

    def _analyze_performance_bottlenecks(self) -> List[Dict[str, Any]]:
        """Analyze which components are causing the most performance issues."""
        if not self.debate_timings:
            return []

        # Find persona performance outliers (top 20% of slowest performers)
        slow_personas = []
        for persona, performances in self.persona_performance.items():
            if len(performances) > 0:
                avg_persona_duration = sum(p["duration"] for p in performances) / len(
                    performances
                )
                avg_persona_tokens = sum(p["tokens_used"] for p in performances) / len(
                    performances
                )

                slow_personas.append(
                    {
                        "persona": persona,
                        "avg_duration": avg_persona_duration,
                        "avg_tokens": avg_persona_tokens,
                        "call_count": len(performances),
                        "success_rate": sum(1 for p in performances if p["success"])
                        / len(performances),
                    }
                )

        # Sort by duration and return top 20% (or top 3, whichever is greater)
        slow_personas.sort(key=lambda x: x["avg_duration"], reverse=True)
        top_count = max(1, len(slow_personas) // 5)  # Top 20%
        return slow_personas[:top_count]

    def _analyze_token_efficiency(self) -> Dict[str, Any]:
        """Analyze token usage patterns to identify inefficiencies."""
        token_efficiency = {
            "total_tokens_used": 0,
            "avg_tokens_per_debate": 0,
            "tokens_per_second": 0,
            "most_inefficient_personas": [],
        }

        # Aggregate token usage across all personas
        total_tokens = 0
        total_calls = 0
        persona_tokens = defaultdict(int)
        persona_calls = defaultdict(int)

        for persona, performances in self.persona_performance.items():
            persona_total = sum(p["tokens_used"] for p in performances)
            persona_tokens[persona] = persona_total
            persona_calls[persona] = len(performances)
            total_tokens += persona_total
            total_calls += len(performances)

        token_efficiency["total_tokens_used"] = total_tokens

        if total_calls > 0:
            token_efficiency["avg_tokens_per_debate"] = (
                total_tokens / len(self.debate_timings) if self.debate_timings else 0
            )
            token_efficiency["tokens_per_second"] = (
                total_tokens / sum(self.debate_timings) if self.debate_timings else 0
            )

        # Identify most token-intensive personas (top 20%)
        sorted_personas = sorted(
            persona_tokens.items(), key=lambda x: x[1], reverse=True
        )
        top_token_personas = sorted_personas[
            : max(1, len(sorted_personas) // 5)
        ]  # Top 20%

        token_efficiency["most_inefficient_personas"] = [
            {
                "persona": persona,
                "total_tokens": tokens,
                "call_count": persona_calls[persona],
                "avg_tokens_per_call": tokens / persona_calls[persona],
            }
            for persona, tokens in top_token_personas
        ]

        return token_efficiency

    def _analyze_error_patterns(self) -> Dict[str, Any]:
        """Analyze error patterns to identify the most common issues."""
        if not self.error_counts:
            return {"error_counts": {}, "top_errors": []}

        # Sort errors by frequency (top 20% or top 3, whichever is greater)
        sorted_errors = sorted(
            self.error_counts.items(), key=lambda x: x[1], reverse=True
        )
        top_count = max(1, len(sorted_errors) // 5)  # Top 20%

        return {
            "error_counts": dict(self.error_counts),
            "top_errors": [
                {"error_type": err[0], "count": err[1]}
                for err in sorted_errors[:top_count]
            ],
        }

    def _analyze_resource_usage(self) -> Dict[str, Any]:
        """Analyze resource usage patterns."""
        try:
            # Get current resource usage
            current_cpu = self.process.cpu_percent()
            current_memory_mb = self.process.memory_info().rss / 1024 / 1024

            # Find recent resource usage metrics
            cpu_metrics = [
                m
                for m in self.metrics_history
                if m.metric_name == "cpu_percent" and "system" in m.tags
            ]
            memory_metrics = [
                m
                for m in self.metrics_history
                if m.metric_name == "memory_rss_mb" and "system" in m.tags
            ]

            if cpu_metrics:
                avg_cpu = sum(
                    m.value for m in cpu_metrics[-MIN_METRICS_FOR_AVERAGE:]
                ) / len(cpu_metrics[-MIN_METRICS_FOR_AVERAGE:])  # Last 20 readings
            else:
                avg_cpu = current_cpu

            if memory_metrics:
                avg_memory = sum(
                    m.value for m in memory_metrics[-MIN_METRICS_FOR_AVERAGE:]
                ) / len(memory_metrics[-MIN_METRICS_FOR_AVERAGE:])  # Last 20 readings
            else:
                avg_memory = current_memory_mb

            return {
                "current_cpu_percent": current_cpu,
                "current_memory_mb": current_memory_mb,
                "avg_cpu_percent": avg_cpu,
                "avg_memory_mb": avg_memory,
                "memory_trend": "increasing"
                if len(memory_metrics) >= MIN_MEMORY_METRICS_FOR_TREND
                and memory_metrics[-1].value
                > memory_metrics[-MIN_MEMORY_METRICS_FOR_TREND].value
                else "stable",
            }
        except Exception:
            return {"error": "Could not collect resource metrics"}

    def get_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report of the most important metrics."""
        return {
            "timestamp": datetime.now().isoformat(),
            "debate_stats": {
                "total_debates": len(self.debate_timings),
                "avg_duration": sum(self.debate_timings) / len(self.debate_timings)
                if self.debate_timings
                else 0,
                "total_duration": sum(self.debate_timings),
            },
            "persona_stats": {
                "total_personas": len(self.persona_performance),
                "total_turns": sum(
                    len(performances)
                    for performances in self.persona_performance.values()
                ),
                "avg_success_rate": self._calculate_avg_success_rate(),
            },
            "error_stats": {
                "total_errors": sum(self.error_counts.values()),
                "error_types": list(self.error_counts.keys()),
            },
            "pareto_analysis": self.get_pareto_analysis(),
        }

    def _calculate_avg_success_rate(self) -> float:
        """Calculate overall success rate across all personas."""
        total_turns = 0
        successful_turns = 0

        for performances in self.persona_performance.values():
            total_turns += len(performances)
            successful_turns += sum(1 for p in performances if p["success"])

        return successful_turns / total_turns if total_turns > 0 else 0.0

    def reset_metrics(self):
        """Reset all collected metrics (preserving configuration)."""
        self.metrics_history.clear()
        self.current_metrics.clear()
        self.debate_timings.clear()
        self.persona_performance.clear()
        self.error_counts.clear()


# Global system monitor instance
system_monitor = SystemMonitor()


def get_system_monitor() -> SystemMonitor:
    """Get the global system monitor instance."""
    return system_monitor
