"""
Dedicated monitoring application for Project Chimera.
Provides real-time metrics and dashboards following the 80/20 Pareto principle.
"""

import streamlit as st

from src.monitoring.dashboard import display_monitoring_dashboard
from src.monitoring.pareto_optimizer import get_pareto_optimizer
from src.monitoring.system_monitor import get_system_monitor

# Set page config
st.set_page_config(
    page_title="Project Chimera - Monitoring Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Start monitoring if not already started
monitor = get_system_monitor()
if not monitor.monitoring_active:
    monitor.start_monitoring()

# Main header
st.title("📊 Project Chimera - Monitoring Dashboard")
st.markdown("""
Following the **80/20 Pareto principle**, this dashboard shows the most impactful metrics
that drive 80% of system performance and efficiency gains.
""")

# Display the monitoring dashboard
display_monitoring_dashboard()

# Add optimization report section
st.header("🎯 80/20 Optimization Report")
try:
    optimizer = get_pareto_optimizer()
    report = optimizer.get_optimization_report()

    if report["summary"]["total_recommendations"] > 0:
        col1, col2, col3, col4 = st.columns(4)
        summary = report["summary"]

        with col1:
            st.metric("Total Recommendations", summary["total_recommendations"])
        with col2:
            st.metric("High Priority Items", summary["high_priority_recommendations"])
        with col3:
            st.metric("Time Savings Potential", summary["potential_time_savings_per_debate"])
        with col4:
            st.metric("Cost Savings Potential", summary["potential_cost_savings_usd"])

        # Show top recommendations
        if report["recommendations"]:
            st.subheader("Top Optimization Recommendations")
            for rec in report["recommendations"][:5]:  # Show top 5
                with st.expander(f"**{rec['title']}** - {rec['priority'].upper()}", expanded=False):
                    st.write(f"**Component:** {rec['component']}")
                    st.write(f"**Impact:** {rec['impact_percentage']:.0f}%")
                    st.write(f"**Effort:** {rec['effort_level'].capitalize()}")
                    st.write(f"**Timeline:** {rec['expected_timeline'].capitalize()}")
                    st.write(f"**Confidence:** {rec['confidence_level']:.0%}")
                    st.write(f"**Description:** {rec['description']}")
    else:
        st.info("Run some debates to generate optimization recommendations based on actual performance data.")
except Exception as e:
    st.error(f"Error loading optimization report: {str(e)}")

# Add some additional monitoring controls
with st.sidebar:
    st.header("Monitoring Controls")

    # Start/stop monitoring
    if st.button("Start System Monitoring") and not monitor.monitoring_active:
        monitor.start_monitoring()
        st.success("System monitoring started!")

    if st.button("Stop System Monitoring") and monitor.monitoring_active:
        monitor.stop_monitoring()
        st.info("System monitoring stopped.")

    # Reset metrics
    if st.button("Reset Metrics"):
        monitor.reset_metrics()
        st.success("Metrics reset!")

    # Optimization controls
    st.subheader("Optimization")
    optimizer = get_pareto_optimizer()
    if st.button("Refresh Optimizations"):
        optimizer.generate_optimizations(force_refresh=True)
        st.success("Optimizations refreshed!")

    # Show current monitoring status
    st.subheader("Status")
    st.write(f"Monitoring Active: {'✅' if monitor.monitoring_active else '❌'}")

    # Show some quick stats
    st.subheader("Quick Stats")
    summary = monitor.get_summary_report()
    st.metric("Total Debates", summary["debate_stats"]["total_debates"])
    st.metric("Avg Duration", f"{summary['debate_stats']['avg_duration']:.2f}s")
    st.metric("Success Rate", f"{summary['persona_stats']['avg_success_rate']:.1%}")
    st.metric("Total Errors", summary["error_stats"]["total_errors"])

# Footer
st.markdown("---")
st.markdown(
    "*Project Chimera - AI System for Socratic Self-Debate and Continuous Improvement*"
)
