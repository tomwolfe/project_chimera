#!/usr/bin/env python3
"""
Automated 80/20 Pareto Analysis Script for Project Chimera
This script runs an analysis on the system's performance and provides optimization recommendations.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.monitoring.pareto_optimizer import get_pareto_optimizer
from src.monitoring.system_monitor import get_system_monitor


def run_80_20_analysis(output_format="console"):
    """Run 80/20 analysis and return results."""
    print("🔍 Running 80/20 Pareto Analysis for Project Chimera...")

    # Get system monitor and optimizer
    monitor = get_system_monitor()
    optimizer = get_pareto_optimizer()

    # Generate recommendations
    recommendations = optimizer.generate_optimizations(force_refresh=True)

    # Get full report
    report = optimizer.get_optimization_report()

    if output_format == "console":
        print("\n" + "="*80)
        print("📊 PROJECT CHIMERA - 80/20 PARETO ANALYSIS REPORT")
        print("="*80)

        print(f"\n📋 Report Generated: {report['timestamp']}")
        print(f"🎯 Focus: {report['focus_area']}")

        print("\n📈 SUMMARY STATISTICS:")
        summary = report['summary']
        print(f"   • Total Recommendations: {summary['total_recommendations']}")
        print(f"   • High Priority Items: {summary['high_priority_recommendations']}")
        print(f"   • Potential Time Savings: {summary['potential_time_savings_per_debate']}")
        print(f"   • Potential Cost Savings: {summary['potential_cost_savings_usd']}")

        print("\n💡 TOP RECOMMENDATIONS (80/20 Focus):")
        for i, rec in enumerate(report['recommendations'][:5], 1):
            print(f"\n   {i}. {rec['title']}")
            print(f"      Component: {rec['component']}")
            print(f"      Priority: {rec['priority'].upper()}")
            print(f"      Impact: {rec['impact_percentage']:.0f}%")
            print(f"      Effort: {rec['effort_level'].capitalize()}")
            print(f"      Timeline: {rec['expected_timeline'].capitalize()}")
            print(f"      Description: {rec['description']}")

        print("\n" + "="*80)
        print("💡 IMPLEMENTATION SUGGESTIONS:")
        print("   1. Focus on HIGH priority items first for maximum impact")
        print("   2. Start with LOW effort items for quick wins")
        print("   3. Monitor metrics after each optimization to measure impact")
        print("="*80)

    elif output_format == "json":
        # Output in JSON format
        print(json.dumps(report, indent=2))

    elif output_format == "markdown":
        # Output in Markdown format
        md_output = f"""# Project Chimera - 80/20 Pareto Analysis Report

Generated: {report['timestamp']}

## Summary
- Total Recommendations: {summary['total_recommendations']}
- High Priority Items: {summary['high_priority_recommendations']}
- Potential Time Savings: {summary['potential_time_savings_per_debate']}
- Potential Cost Savings: {summary['potential_cost_savings_usd']}

## Top Recommendations (80/20 Focus)

"""
        for i, rec in enumerate(report['recommendations'][:5], 1):
            md_output += f"""### {i}. {rec['title']}
- **Component**: {rec['component']}
- **Priority**: {rec['priority'].upper()}
- **Impact**: {rec['impact_percentage']:.0f}%
- **Effort**: {rec['effort_level'].capitalize()}
- **Timeline**: {rec['expected_timeline'].capitalize()}
- **Description**: {rec['description']}

"""

        md_output += """## Implementation Suggestions
1. Focus on HIGH priority items first for maximum impact
2. Start with LOW effort items for quick wins
3. Monitor metrics after each optimization to measure impact
"""
        print(md_output)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Run 80/20 Pareto Analysis for Project Chimera"
    )
    parser.add_argument(
        "--format",
        choices=["console", "json", "markdown"],
        default="console",
        help="Output format (default: console)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="File to save the report (optional)"
    )

    args = parser.parse_args()

    # Run the analysis
    report = run_80_20_analysis(output_format=args.format)

    # Save to file if requested
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            if args.format == "json":
                json.dump(report, f, indent=2)
            else:
                # Convert to the requested format for file output
                temp_report = run_80_20_analysis(output_format="json")
                json.dump(temp_report, f, indent=2)
        print(f"\n💾 Report saved to: {args.output_file}")


if __name__ == "__main__":
    main()
