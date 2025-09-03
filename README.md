# Project Chimera

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://python.org)
[![GitHub stars](https://img.shields.io/github/stars/tomwolfe/project_chimera?style=social)](https://github.com/tomwolfe/project_chimera)

**Project Chimera is an advanced reasoning engine that critically analyzes its own codebase to identify and implement self-improvements through Socratic self-debate methodology.** This innovative system employs multiple specialized AI personas that engage in structured debate to produce higher quality outputs while continuously optimizing its own capabilities.

## 🌟 Overview

Project Chimera represents a paradigm shift in AI-assisted software development. Unlike traditional tools, Chimera doesn't just generate code—it actively improves itself through a continuous cycle of self-analysis and refinement. The system employs multiple specialized AI personas that engage in Socratic debate to produce higher quality outputs while identifying opportunities for its own enhancement.

Key differentiators:
- **True self-improvement capability**: Analyzes its own codebase to identify and implement optimizations
- **80/20 Pareto principle focus**: Prioritizes the most impactful improvements first
- **Multi-persona architecture**: Specialized AI roles collaborate through structured debate
- **Metrics-driven optimization**: Tracks reasoning quality, robustness, efficiency, and maintainability

## 🔑 Key Features

- **Socratic Self-Debate Methodology**: Multiple AI personas collaborate through structured debate to refine solutions
- **Self-Improvement Loop**: Automatically identifies high-impact code improvements in its own implementation
- **Token Optimization**: Intelligent management of LLM token usage while maintaining output quality
- **Multi-Domain Expertise**: Specialized personas for coding, security, scientific analysis, and self-improvement
- **Comprehensive Metrics Tracking**: Monitors reasoning quality, robustness, efficiency, maintainability, security, test coverage, CI/CD, and token usage
- **Automated Feedback Integration**: Captures schema validation failures and content misalignment warnings to drive continuous improvement

## 🏗️ Architecture Overview

Project Chimera follows a sophisticated multi-layered architecture:

- **Core Engine**: Manages the Socratic debate process and coordinates persona interactions
- **LLM Interface**: Provides abstraction for different LLM providers (currently Gemini)
- **Self-Improvement Module**: 
  - `FocusedMetricsCollector`: Gathers objective metrics for improvement targeting
  - `ContentValidator`: Ensures output alignment with requirements
  - `TokenUsageTracker`: Monitors and optimizes token consumption
  - `AutoRemediator`: Automatically applies safe, well-defined fixes
- **Persona Management**: Coordinates specialized AI roles with distinct expertise
- **Prompt Engineering System**: Dynamically optimizes prompts for efficiency and effectiveness

![Architecture Diagram](https://i.imgur.com/architecture-diagram.png)

## ⚙️ Technical Stack

- **Language**: Python 3.11+
- **LLM Provider**: Google Generative AI API (Gemini)
- **UI Framework**: Streamlit
- **Code Quality**: Ruff (linting and formatting)
- **Security**: Bandit (security scanning)
- **Testing**: pytest
- **CI/CD**: GitHub Actions
- **Key Components**:
  - `SocraticDebate` engine
  - `SelfImprovementLoop` manager
  - `PersonaManager` for role coordination
  - `PromptAnalyzer` for dynamic prompt optimization
  - `FocusedMetricsCollector` for targeted improvement analysis

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Google API key for Gemini access

### Installation
```bash
# Clone the repository
git clone https://github.com/tomwolfe/project_chimera.git
cd project_chimera

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export GEMINI_API_KEY=your_api_key

# Run the application
streamlit run app.py
```

### Configuration
The system is configured through `pyproject.toml` which manages:
- Ruff (linting and formatting)
- Bandit (security scanning)
- Other project settings

## 🧠 Self-Improvement Methodology

Project Chimera follows a rigorous, phased approach to self-optimization:

### 1. Enhanced Observability & Instrumentation
- Comprehensive metrics collection across 8 focus areas:
  - Reasoning quality
  - Robustness
  - Efficiency
  - Developer maintainability
  - Security
  - Test coverage
  - CI/CD
  - Token usage
- System state monitoring
- Performance baseline establishment

### 2. Simulation & Validation
- Sandboxed testing environment
- Automated validation of proposed changes
- Impact analysis on reasoning quality and robustness

### 3. Iterative, Validated Code Modifications
- Targeted improvements based on data
- Adherence to the 80/20 Pareto principle
- Continuous validation of enhancements

All improvements prioritize the core focus areas in this order:
1. Reasoning quality
2. Robustness
3. Efficiency
4. Developer maintainability
5. Security
6. Test coverage
7. CI/CD
8. Token usage

### 4. Integrating Feedback into the Loop
- **Automated Feedback**: System automatically captures schema validation failures and content misalignment warnings
- **Self-Correction Mechanism**: 
  - Triggers re-analysis of problematic output with refined prompts
  - Prioritizes prompt adjustments for personas that consistently fail validation
  - Creates specific test cases to address recurring alignment issues

## 🤝 Contributing

We welcome contributions to Project Chimera! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

When contributing self-improvement suggestions, please:
- Focus on the 80/20 Pareto principle
- Provide actionable code modifications
- Reference relevant metrics
- Prioritize reasoning quality, robustness, efficiency, and maintainability

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📱 Connect With Us

*   **GitHub**: [tomwolfe/project_chimera](https://github.com/tomwolfe/project_chimera)
*   **Twitter**: [@Proj_Chimera](https://x.com/Proj_Chimera)

## 🙏 Acknowledgments

Project Chimera builds upon numerous open-source projects and frameworks. Special thanks to the Python, Streamlit, Bandit, Ruff, and GitHub Actions communities for their excellent tools and libraries.

---

> "The only true wisdom is in knowing you know nothing." - Socrates  
> Project Chimera embodies this philosophy through continuous self-reflection and improvement.