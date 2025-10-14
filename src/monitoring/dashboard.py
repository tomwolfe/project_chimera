"""
Monitoring dashboard for Project Chimera following the 80/20 Pareto principle.
Provides visualizations for the most impactful metrics.
"""

from datetime import datetime
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.monitoring.pareto_optimizer import get_pareto_optimizer
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


def display_performance_metrics(monitor, pareto_analysis: dict[str, Any]):
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


def display_token_metrics(monitor, pareto_analysis: dict[str, Any]):
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


def display_error_metrics(monitor, pareto_analysis: dict[str, Any]):
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


def display_resource_metrics(monitor, pareto_analysis: dict[str, Any]):
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


def display_optimization_recommendations():
    """Display optimization recommendations based on 80/20 analysis."""
    st.subheader("💡 Optimization Recommendations (80/20 Focus)")

    try:
        optimizer = get_pareto_optimizer()
        recommendations = optimizer.generate_optimizations()

        if recommendations:
            # Group recommendations by priority
            high_priority = [
                r for r in recommendations if r.priority in ["critical", "high"]
            ]
            medium_priority = [r for r in recommendations if r.priority == "medium"]

            # Display high priority recommendations first
            if high_priority:
                st.markdown("#### 🔥 High Priority Optimizations")
                for rec in high_priority[:5]:  # Show top 5
                    with st.expander(
                        f"**{rec.title}** (Impact: {rec.impact_percentage:.0f}%)",
                        expanded=False,
                    ):
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.write(f"**Component:** {rec.component}")
                            st.write(f"**Description:** {rec.description}")
                            st.write(
                                f"**Effort Level:** {rec.effort_level.capitalize()}"
                            )
                        with col2:
                            st.metric("Impact", f"{rec.impact_percentage:.0f}%")
                            st.metric("Timeline", rec.expected_timeline.capitalize())

                        if rec.estimated_savings:
                            st.write("**Estimated Savings:**")
                            savings_items = []
                            for key, value in rec.estimated_savings.items():
                                if key != "cost_usd" and isinstance(
                                    value, (int, float)
                                ):
                                    savings_items.append(
                                        f"{key}: {value:.2f}"
                                        if isinstance(value, float)
                                        else f"{key}: {value:,}"
                                    )
                                elif key == "cost_usd" and isinstance(value, float):
                                    savings_items.append(f"Cost: ${value:.4f}")
                            if savings_items:
                                st.write(", ".join(savings_items))

                        st.write("**Implementation Steps:**")
                        for step in rec.implementation_steps[:3]:  # Show first 3 steps
                            st.write(f"- {step}")

            # Medium priority
            if medium_priority:
                st.markdown("#### ⚡ Medium Priority Optimizations")
                for rec in medium_priority[:5]:  # Show top 5
                    with st.expander(
                        f"{rec.title} (Impact: {rec.impact_percentage:.0f}%)",
                        expanded=False,
                    ):
                        st.write(f"**Component:** {rec.component}")
                        st.write(f"Description: {rec.description}")
                        st.write(f"Effort Level: {rec.effort_level.capitalize()}")

            # Summary statistics
            if recommendations:
                st.markdown("#### 📊 Optimization Summary")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Recommendations", len(recommendations))
                with col2:
                    st.metric("High Priority", len(high_priority))
                with col3:
                    total_potential_time = sum(
                        rec.estimated_savings.get("time", 0) for rec in recommendations
                    )
                    st.metric("Potential Time Savings", f"{total_potential_time:.1f}s")
                with col4:
                    total_potential_cost = sum(
                        rec.estimated_savings.get("cost_usd", 0)
                        for rec in recommendations
                    )
                    st.metric("Potential Cost Savings", f"${total_potential_cost:.4f}")
        else:
            st.info(
                "No optimization recommendations available. Run a few debates to gather performance data."
            )

    except Exception as e:
        st.error(f"Error loading optimization recommendations: {str(e)}")


def display_pareto_charts(pareto_analysis: dict[str, Any]):
    """Display Pareto charts to visualize the 80/20 principle."""
    st.subheader("📊 80/20 Pareto Analysis Charts")

    # Performance bottlenecks chart
    bottlenecks = pareto_analysis.get("performance_bottlenecks", [])
    if bottlenecks:
        st.markdown("### Top Performance Bottlenecks (20% causing 80% of delays)")
        bottleneck_df = pd.DataFrame(bottlenecks)
        if not bottleneck_df.empty:
            # Create a Pareto chart for performance bottlenecks
            bottleneck_df = bottleneck_df.sort_values("avg_duration", ascending=False)
            bottleneck_df["cumulative_percentage"] = (
                bottleneck_df["avg_duration"].cumsum()
                / bottleneck_df["avg_duration"].sum()
            ) * 100

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=bottleneck_df["persona"],
                    y=bottleneck_df["avg_duration"],
                    name="Avg Duration",
                    marker_color="red",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=bottleneck_df["persona"],
                    y=bottleneck_df["cumulative_percentage"],
                    mode="lines+markers",
                    name="Cumulative %",
                    yaxis="y2",
                    line=dict(color="blue", dash="dash"),
                    marker=dict(color="blue"),
                )
            )

            fig.update_layout(
                title="Performance Bottlenecks - Pareto Analysis",
                xaxis_title="Persona",
                yaxis_title="Avg Duration (s)",
                yaxis2=dict(title="Cumulative %", overlaying="y", side="right"),
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Token inefficiency chart
    token_analysis_pareto = pareto_analysis.get("token_efficiency", {})
    inefficient_personas = token_analysis_pareto.get("most_inefficient_personas", [])
    if inefficient_personas:
        st.markdown("### Most Token-Inefficient Personas (20% using 80% of tokens)")
        token_df = pd.DataFrame(inefficient_personas)
        if not token_df.empty:
            # Create a Pareto chart for token inefficiency
            token_df = token_df.sort_values("total_tokens", ascending=False)
            token_df["cumulative_percentage"] = (
                token_df["total_tokens"].cumsum() / token_df["total_tokens"].sum()
            ) * 100

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=token_df["persona"],
                    y=token_df["total_tokens"],
                    name="Total Tokens",
                    marker_color="orange",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=token_df["persona"],
                    y=token_df["cumulative_percentage"],
                    mode="lines+markers",
                    name="Cumulative %",
                    yaxis="y2",
                    line=dict(color="blue", dash="dash"),
                    marker=dict(color="blue"),
                )
            )

            fig.update_layout(
                title="Token Usage - Pareto Analysis",
                xaxis_title="Persona",
                yaxis_title="Total Tokens",
                yaxis2=dict(title="Cumulative %", overlaying="y", side="right"),
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)


def display_system_overview(summary: dict[str, Any], pareto_analysis: dict[str, Any]):
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

    # Add the interactive Pareto charts
    display_pareto_charts(pareto_analysis)

    # Add optimization recommendations
    display_optimization_recommendations()


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
