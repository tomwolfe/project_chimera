"""
Monitoring dashboard for Project Chimera following the 80/20 Pareto principle.
Provides visualizations for the most impactful metrics.
"""

from datetime import datetime
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import streamlit as st

from src.monitoring.system_monitor import MetricType, get_system_monitor


def display_monitoring_dashboard():
    """Display the main monitoring dashboard with key metrics and visualizations."""
    monitor = get_system_monitor()

    st.header("📊 System Monitoring Dashboard")
    st.markdown(
        "Following the 80/20 Pareto principle - tracking the most impactful metrics"
    )

    # Get summary report
    summary = monitor.get_summary_report()
    pareto_analysis = summary.get("pareto_analysis", {})

    # Create tabs for different monitoring views
    tabs = st.tabs(
        ["System Overview", "Performance", "Token Usage", "Errors", "Resource Usage"]
    )

    with tabs[0]:  # System Overview
        display_system_overview(summary, pareto_analysis)

    with tabs[1]:  # Performance
        display_performance_metrics(monitor, pareto_analysis)

    with tabs[2]:  # Token Usage
        display_token_metrics(monitor, pareto_analysis)

    with tabs[3]:  # Errors
        display_error_metrics(monitor, pareto_analysis)

    with tabs[4]:  # Resource Usage
        display_resource_metrics(monitor, pareto_analysis)

    # Auto-refresh option
    if st.checkbox("Auto-refresh every 30 seconds"):
        st.experimental_rerun()


def display_system_overview(summary: Dict[str, Any], pareto_analysis: Dict[str, Any]):
    """Display the system overview with key summary metrics."""
    st.subheader("System Overview")

    # Create summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Total Debates", value=summary["debate_stats"]["total_debates"])

    with col2:
        avg_duration = summary["debate_stats"]["avg_duration"]
        st.metric(label="Avg Duration (s)", value=f"{avg_duration:.2f}")

    with col3:
        avg_success_rate = summary["persona_stats"]["avg_success_rate"]
        st.metric(label="Success Rate", value=f"{avg_success_rate:.1%}")

    with col4:
        total_errors = summary["error_stats"]["total_errors"]
        st.metric(label="Total Errors", value=total_errors)

    # Display Pareto analysis findings
    st.subheader("80/20 Analysis Findings")

    # Performance bottlenecks
    bottlenecks = pareto_analysis.get("performance_bottlenecks", [])
    if bottlenecks:
        st.markdown("### Top Performance Bottlenecks (20% causing 80% of delays)")
        bottleneck_df = pd.DataFrame(bottlenecks)
        st.dataframe(bottleneck_df, use_container_width=True)

    # Most inefficient personas
    token_analysis = pareto_analysis.get("token_efficiency", {})
    inefficient_personas = token_analysis.get("most_inefficient_personas", [])
    if inefficient_personas:
        st.markdown("### Most Token-Inefficient Personas (20% using 80% of tokens)")
        token_df = pd.DataFrame(inefficient_personas)
        st.dataframe(token_df, use_container_width=True)


def display_performance_metrics(monitor, pareto_analysis: Dict[str, Any]):
    """Display performance-specific metrics and visualizations."""
    st.subheader("Performance Metrics")

    # Debate duration over time
    if monitor.debate_timings:
        debate_timeseries = pd.DataFrame(
            {
                "debate": range(len(monitor.debate_timings)),
                "duration": monitor.debate_timings,
            }
        )

        fig = px.line(
            debate_timeseries,
            x="debate",
            y="duration",
            title="Debate Duration Over Time",
            labels={"duration": "Duration (seconds)", "debate": "Debate #"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Persona performance heatmap
    if monitor.persona_performance:
        personas = list(monitor.persona_performance.keys())
        if personas:
            performance_data = []
            for persona in personas:
                performances = monitor.persona_performance[persona]
                if performances:
                    avg_duration = sum(p["duration"] for p in performances) / len(
                        performances
                    )
                    success_rate = sum(1 for p in performances if p["success"]) / len(
                        performances
                    )
                    avg_tokens = sum(p["tokens_used"] for p in performances) / len(
                        performances
                    )

                    performance_data.append(
                        {
                            "Persona": persona,
                            "Avg Duration": avg_duration,
                            "Success Rate": success_rate,
                            "Avg Tokens": avg_tokens,
                            "Call Count": len(performances),
                        }
                    )

            if performance_data:
                perf_df = pd.DataFrame(performance_data)
                fig = px.scatter(
                    perf_df,
                    x="Avg Duration",
                    y="Success Rate",
                    size="Avg Tokens",
                    color="Call Count",
                    hover_data=["Persona"],
                    title="Persona Performance: Duration vs Success Rate",
                )
                st.plotly_chart(fig, use_container_width=True)


def display_token_metrics(monitor, pareto_analysis: Dict[str, Any]):
    """Display token usage metrics and visualizations."""
    st.subheader("Token Usage Metrics")

    token_analysis = pareto_analysis.get("token_efficiency", {})

    # Token efficiency summary
    if token_analysis:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="Total Tokens Used",
                value=f"{token_analysis.get('total_tokens_used', 0):,}",
            )
        with col2:
            st.metric(
                label="Avg Tokens/Debate",
                value=f"{token_analysis.get('avg_tokens_per_debate', 0):.0f}",
            )
        with col3:
            st.metric(
                label="Tokens/Second",
                value=f"{token_analysis.get('tokens_per_second', 0):.1f}",
            )

    # Token usage by persona
    if monitor.persona_performance:
        token_by_persona = []
        for persona, performances in monitor.persona_performance.items():
            total_tokens = sum(p["tokens_used"] for p in performances)
            call_count = len(performances)
            if total_tokens > 0:
                token_by_persona.append(
                    {
                        "Persona": persona,
                        "Total Tokens": total_tokens,
                        "Call Count": call_count,
                        "Avg Tokens": total_tokens / call_count,
                    }
                )

        if token_by_persona:
            token_df = pd.DataFrame(token_by_persona)
            token_df = token_df.sort_values("Total Tokens", ascending=False)

            fig = px.bar(
                token_df,
                x="Persona",
                y="Total Tokens",
                title="Token Usage by Persona (Top Consumers)",
                text="Total Tokens",
            )
            fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)


def display_error_metrics(monitor, pareto_analysis: Dict[str, Any]):
    """Display error metrics and visualizations."""
    st.subheader("Error Metrics")

    error_analysis = pareto_analysis.get("error_patterns", {})

    if error_analysis and error_analysis.get("top_errors"):
        # Top errors by count
        top_errors = error_analysis["top_errors"]
        error_df = pd.DataFrame(top_errors)

        fig = px.bar(
            error_df,
            x="error_type",
            y="count",
            title="Top Error Types (80/20)",
            text="count",
        )
        fig.update_traces(texttemplate="%{text}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # Error timeline (if we have timestamped error data)
    error_metrics = [
        m for m in monitor.metrics_history if m.metric_type == MetricType.ERROR_RATE
    ]

    if error_metrics:
        error_timeline = pd.DataFrame(
            [
                {
                    "timestamp": m.timestamp,
                    "error_type": m.metric_name.replace("error_", ""),
                    "value": m.value,
                }
                for m in error_metrics
            ]
        )

        if not error_timeline.empty:
            fig = px.line(
                error_timeline,
                x="timestamp",
                y="value",
                color="error_type",
                title="Error Occurrences Over Time",
            )
            st.plotly_chart(fig, use_container_width=True)


def display_resource_metrics(monitor, pareto_analysis: Dict[str, Any]):
    """Display resource usage metrics."""
    st.subheader("Resource Usage")

    resource_analysis = pareto_analysis.get("resource_usage", {})

    if resource_analysis:
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Current CPU %",
                value=f"{resource_analysis.get('current_cpu_percent', 0):.1f}",
            )
        with col2:
            st.metric(
                label="Current Memory (MB)",
                value=f"{resource_analysis.get('current_memory_mb', 0):.1f}",
            )

        col3, col4 = st.columns(2)
        with col3:
            st.metric(
                label="Avg CPU %",
                value=f"{resource_analysis.get('avg_cpu_percent', 0):.1f}",
            )
        with col4:
            st.metric(
                label="Avg Memory (MB)",
                value=f"{resource_analysis.get('avg_memory_mb', 0):.1f}",
            )

    # Resource usage timeline
    resource_metrics = [
        m
        for m in monitor.metrics_history
        if m.metric_type == MetricType.RESOURCE_UTILIZATION
    ]

    if resource_metrics:
        resource_data = []
        for metric in resource_metrics:
            resource_data.append(
                {
                    "timestamp": metric.timestamp,
                    "metric": metric.metric_name,
                    "value": metric.value,
                    "tags": metric.tags,
                }
            )

        if resource_data:
            resource_df = pd.DataFrame(resource_data)

            # Plot CPU usage
            cpu_data = resource_df[resource_df["metric"] == "cpu_percent"]
            if not cpu_data.empty:
                fig = px.line(
                    cpu_data, x="timestamp", y="value", title="CPU Usage Over Time"
                )
                fig.update_layout(yaxis_title="CPU %")
                st.plotly_chart(fig, use_container_width=True)

            # Plot Memory usage
            memory_data = resource_df[resource_df["metric"] == "memory_rss_mb"]
            if not memory_data.empty:
                fig = px.line(
                    memory_data,
                    x="timestamp",
                    y="value",
                    title="Memory Usage Over Time",
                )
                fig.update_layout(yaxis_title="Memory (MB)")
                st.plotly_chart(fig, use_container_width=True)


def add_monitoring_to_debate_process():
    """Decorator or utility function to add monitoring to debate processes."""

    def monitoring_decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_system_monitor()
            start_time = datetime.now()

            try:
                result = func(*args, **kwargs)

                # Record success metrics
                duration = (datetime.now() - start_time).total_seconds()
                monitor.record_debate_timing(duration)

                return result
            except Exception as e:
                # Record error
                duration = (datetime.now() - start_time).total_seconds()
                monitor.record_debate_timing(duration)
                monitor.record_error(type(e).__name__, str(e))
                raise

        return wrapper

    return monitoring_decorator
