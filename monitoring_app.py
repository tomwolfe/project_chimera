"""
Dedicated monitoring application for Project Chimera.
Provides real-time metrics and dashboards following the 80/20 Pareto principle.
"""

import streamlit as st

from src.monitoring.dashboard import display_monitoring_dashboard
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
