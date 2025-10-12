"""
Enhanced logging module for Project Chimera following the 80/20 Pareto principle.
Improves tracking of performance metrics and errors for maximum impact.
"""

import logging
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pythonjsonlogger import jsonlogger

from src.monitoring.system_monitor import get_system_monitor


class LogLevel(Enum):
    """Enhanced log levels for better categorization."""

    PERFORMANCE = "performance"
    METRICS = "metrics"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"
    CRITICAL = "critical"


@dataclass
class LogEntry:
    """Enhanced log entry with additional metadata."""

    timestamp: datetime
    level: LogLevel
    message: str
    module: str
    function: str
    line: int
    extra: Dict[str, Any]
    request_id: Optional[str] = None
    duration: Optional[float] = None
    tokens_used: Optional[int] = None
    persona: Optional[str] = None


class EnhancedJSONFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter that adds 80/20 specific fields."""

    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)

        # Add standard fields
        log_record["timestamp"] = datetime.fromtimestamp(record.created).isoformat()
        log_record["level"] = record.levelname
        log_record["module"] = record.module
        log_record["function"] = record.funcName
        log_record["line"] = record.lineno

        # Add 80/20 specific fields
        if hasattr(record, "request_id"):
            log_record["request_id"] = record.request_id
        if hasattr(record, "duration"):
            log_record["duration"] = record.duration
        if hasattr(record, "tokens_used"):
            log_record["tokens_used"] = record.tokens_used
        if hasattr(record, "persona"):
            log_record["persona"] = record.persona
        if hasattr(record, "debate_phase"):
            log_record["debate_phase"] = record.debate_phase


class PerformanceLogger:
    """Enhanced logger focused on performance metrics following the 80/20 principle."""

    def __init__(self, name: str = "chimera_performance"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Create handler with enhanced formatter
        handler = logging.StreamHandler(sys.stdout)
        formatter = EnhancedJSONFormatter(
            "%(timestamp)s %(level)s %(module)s %(function)s %(line)s %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # Disable propagation to avoid duplicate logs
        self.logger.propagate = False

    def log_performance(self, message: str, duration: float, **kwargs):
        """Log performance metrics with 80/20 focus."""
        extra = {"duration": duration, "log_type": "performance"}
        extra.update(kwargs)

        self.logger.info(message, extra=extra)

        # Record metric in system monitor
        from src.monitoring.system_monitor import MetricType

        monitor = get_system_monitor()
        if "request_id" in extra:
            monitor.record_metric(
                MetricType.PERFORMANCE, "operation_duration", duration, context=extra
            )

    def log_token_usage(self, tokens_used: int, tokens_budget: int, **kwargs):
        """Log token usage metrics."""
        usage_percentage = (
            (tokens_used / tokens_budget) * 100 if tokens_budget > 0 else 0
        )
        message = (
            f"Token usage: {tokens_used}/{tokens_budget} ({usage_percentage:.1f}%)"
        )

        extra = {
            "tokens_used": tokens_used,
            "tokens_budget": tokens_budget,
            "usage_percentage": usage_percentage,
            "log_type": "token_usage",
        }
        extra.update(kwargs)

        self.logger.info(message, extra=extra)

        # Record metric in system monitor
        from src.monitoring.system_monitor import MetricType

        monitor = get_system_monitor()
        if "request_id" in extra:
            monitor.record_metric(
                MetricType.TOKEN_USAGE,
                "token_utilization",
                usage_percentage,
                context=extra,
            )

    def log_error(self, error: Exception, context: str = "", **kwargs):
        """Log error with enhanced context for 80/20 analysis."""
        extra = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "traceback": traceback.format_exc(),
            "log_type": "error",
        }
        extra.update(kwargs)

        self.logger.error(f"Error in {context}: {str(error)}", extra=extra)

        # Record error in system monitor
        monitor = get_system_monitor()
        monitor.record_error(type(error).__name__, str(error), extra.get("request_id"))

    def log_persona_performance(
        self, persona: str, success: bool, duration: float, tokens_used: int, **kwargs
    ):
        """Log persona performance metrics."""
        success_rate = "success" if success else "failure"
        message = f"Persona {persona} {success_rate} - Duration: {duration:.2f}s, Tokens: {tokens_used}"

        extra = {
            "persona": persona,
            "success": success,
            "duration": duration,
            "tokens_used": tokens_used,
            "log_type": "persona_performance",
        }
        extra.update(kwargs)

        if success:
            self.logger.info(message, extra=extra)
        else:
            self.logger.warning(message, extra=extra)

        # Record metric in system monitor
        monitor = get_system_monitor()
        if "request_id" in extra:
            monitor.record_persona_performance(
                persona, success, tokens_used, duration, extra["request_id"]
            )


class ParetoAnalyzer:
    """Analyzes logs and metrics to identify 80/20 optimization opportunities."""

    def __init__(self):
        self.performance_data = []
        self.error_data = []
        self.token_data = []

    def analyze_performance_logs(self) -> Dict[str, Any]:
        """Analyze performance logs to find 80/20 optimization opportunities."""
        # This would typically analyze logs from a log file or database
        # For now, we'll use the system monitor data
        monitor = get_system_monitor()
        analysis = monitor.get_pareto_analysis()

        return {
            "timestamp": datetime.now().isoformat(),
            "analysis_type": "performance_80_20",
            "findings": analysis,
        }

    def generate_optimization_recommendations(self) -> Dict[str, Any]:
        """Generate optimization recommendations based on 80/20 analysis."""
        analysis = self.analyze_performance_logs()

        recommendations = {
            "priority_improvements": [],
            "resource_optimizations": [],
            "error_reduction_opportunities": [],
        }

        # Analyze performance bottlenecks
        bottlenecks = analysis["pareto_analysis"]["performance_bottlenecks"]
        for bottleneck in bottlenecks[:2]:  # Top 2 bottlenecks (20% causing 80% issues)
            recommendations["priority_improvements"].append(
                {
                    "component": bottleneck["persona"],
                    "issue": f"High average duration: {bottleneck['avg_duration']:.2f}s",
                    "impact": "Reducing this could improve overall system performance by 20-80%",
                    "suggestion": f"Optimize {bottleneck['persona']} persona logic or reduce its complexity",
                }
            )

        # Analyze token inefficiencies
        token_efficiency = analysis["pareto_analysis"]["token_efficiency"]
        inefficient_personas = token_efficiency.get("most_inefficient_personas", [])
        for persona_info in inefficient_personas[:2]:  # Top 2 inefficient personas
            recommendations["resource_optimizations"].append(
                {
                    "component": persona_info["persona"],
                    "issue": f"High token consumption: {persona_info['total_tokens']:,} tokens",
                    "impact": f"Optimizing this could reduce token costs by {persona_info['avg_tokens_per_call']:.0f} tokens per call",
                    "suggestion": f"Implement more efficient prompts or reduce output length for {persona_info['persona']}",
                }
            )

        # Analyze error patterns
        error_patterns = analysis["pareto_analysis"]["error_patterns"]
        top_errors = error_patterns.get("top_errors", [])
        for error in top_errors[:2]:  # Top 2 error types
            recommendations["error_reduction_opportunities"].append(
                {
                    "error_type": error["error_type"],
                    "frequency": error["count"],
                    "impact": f"Fixing this could eliminate {error['count']} error instances",
                    "suggestion": f"Implement better error handling or validation for {error['error_type']} errors",
                }
            )

        return {
            "timestamp": datetime.now().isoformat(),
            "recommendations": recommendations,
            "analysis_summary": analysis,
        }


# Global performance logger instance
performance_logger = PerformanceLogger()


def get_performance_logger() -> PerformanceLogger:
    """Get the global performance logger instance."""
    return performance_logger


def get_pareto_analyzer() -> ParetoAnalyzer:
    """Get the Pareto analyzer instance."""
    return ParetoAnalyzer()


def performance_timer(func):
    """Decorator to automatically time function execution and log performance."""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        request_id = kwargs.get("request_id") or (
            args[0].request_id if hasattr(args[0], "request_id") else None
        )

        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time

            # Log performance
            extra = {"function": func.__name__, "request_id": request_id}
            perf_logger = get_performance_logger()
            perf_logger.log_performance(
                f"Function {func.__name__} executed successfully",
                duration=duration,
                **extra,
            )

            return result
        except Exception as e:
            duration = time.time() - start_time

            # Log error with performance context
            extra = {
                "function": func.__name__,
                "request_id": request_id,
                "duration": duration,
            }
            perf_logger = get_performance_logger()
            perf_logger.log_error(e, f"Function {func.__name__}", **extra)

            raise

    return wrapper


def log_token_usage(tokens_used: int, tokens_budget: int, request_id: str = None):
    """Convenience function to log token usage."""
    extra = {"request_id": request_id} if request_id else {}
    perf_logger = get_performance_logger()
    perf_logger.log_token_usage(tokens_used, tokens_budget, **extra)


def log_persona_performance(
    persona: str,
    success: bool,
    duration: float,
    tokens_used: int,
    request_id: str = None,
):
    """Convenience function to log persona performance."""
    extra = {"request_id": request_id} if request_id else {}
    perf_logger = get_performance_logger()
    perf_logger.log_persona_performance(
        persona, success, duration, tokens_used, **extra
    )
