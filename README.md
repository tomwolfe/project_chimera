# Project Chimera: Socratic Self-Debate Reasoning Engine

Project Chimera is an advanced reasoning engine designed for complex problem-solving and code generation through Socratic self-debate methodology. This innovative system critically analyzes its own codebase to identify and implement improvements, creating a self-optimizing AI development framework.

*   **GitHub**: [tomwolfe/project_chimera](https://github.com/tomwolfe/project_chimera)
*   **Twitter**: [@Proj_Chimera](https://x.com/Proj_Chimera)

## 🌟 Key Features

- **Socratic Self-Debate Framework**: Multiple specialized AI personas engage in structured debate to solve complex problems
- **Self-Improvement Engine**: Automatically analyzes its own codebase to identify high-impact improvements
- **80/20 Pareto Principle Focus**: Prioritizes changes with the highest impact for the effort required
- **Multi-Persona Architecture**: Includes specialized roles like:
  - Security Auditor (Bandit integration for vulnerability scanning)
  - Code Architect (maintainability and structural improvements)
  - DevOps Engineer (CI/CD and infrastructure optimization)
  - Constructive Critic (identifies flaws in proposed improvements)
  - Self-Improvement Analyst (synthesizes debate results into actionable changes)
  - Impartial Arbitrator (resolves conflicts between personas)
  - Test Engineer (focuses on test coverage and validation)
- **Comprehensive Analysis**: Evaluates code across 8 critical dimensions:
  - Reasoning quality
  - Robustness (schema validation, error handling)
  - Efficiency (token usage optimization)
  - Maintainability (PEP8 compliance, documentation)
  - Security (vulnerability scanning, secret management)
  - Test coverage (unit tests, edge cases)
  - CI/CD processes (Bandit integration, automated workflows)
  - Token usage optimization (persona-specific budgeting)
- **Structured JSON Output**: All self-improvement suggestions follow a strict schema for reliable parsing
- **Circuit Breaker System**: Prevents cascading failures during self-improvement attempts

## 🚀 Getting Started

### Prerequisites
- Python 3.11
- Git

### Installation
```bash
# Clone the repository
git clone https://github.com/tomwolfe/project_chimera.git
cd project_chimera

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/MacOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 🧠 How It Works

Project Chimera employs a unique self-improvement methodology where the system:

1. Analyzes its own codebase using specialized personas with distinct focus areas
2. Each persona generates structured JSON output following strict schema validation
3. An Impartial Arbitrator resolves conflicts between differing persona recommendations
4. A Self-Improvement Analyst synthesizes the debate results into actionable code changes
5. Proposed changes undergo circuit breaker validation before implementation
6. System measures impact of changes against objective metrics for continuous learning

The process strictly follows the 80/20 principle, focusing on the top 20% of changes that deliver 80% of potential benefits.

## 📂 Project Structure

```
project_chimera/
├── .github/                  # CI/CD workflows including Bandit security scanning
│   └── workflows/            # GitHub Actions pipelines
├── src/                      # Core application code
│   ├── core/                 # Main engine and debate framework
│   ├── self_improvement/     # Self-analysis system components
│   │   ├── metrics_collector.py  # Objective metric collection
│   │   └── improvement_loop.py   # Self-improvement orchestration
│   ├── utils/                # Utility functions and helpers
│   │   └── prompt_engineering.py # Dynamic persona routing
│   ├── models/               # Data models and JSON schemas
│   └── config/               # Configuration management
├── tests/                    # Comprehensive test suite
├── .pre-commit-config.yaml   # Includes Bandit security scanning
├── pyproject.toml            # Project configuration
├── requirements.txt          # Development dependencies
└── app.py                    # Streamlit web application entry point
```

## 🔍 Self-Improvement Capabilities

Project Chimera continuously evaluates itself across multiple dimensions:

### Security Analysis
- Automated scanning for vulnerabilities using Bandit
- Hardcoded secrets detection with configuration in pyproject.toml
- Input validation and sanitization checks
- API key management review
- CI/CD pipeline security assessment

### Code Quality & Maintainability
- Ruff-based linting and formatting (PEP8 compliance)
- Technical debt identification
- Code structure analysis (nesting depth, code smells)
- Documentation quality assessment

### Performance & Efficiency
- Token usage optimization across personas
- Response time monitoring
- Resource consumption analysis
- Adaptive persona configuration (temperature, max_tokens)

### Testing & Reliability
- Test coverage analysis
- Edge case identification
- Error handling review
- Circuit breaker implementation for failure prevention

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please review our [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📢 Community

Join our community to discuss features, report bugs, and collaborate:

- **GitHub Discussions**: [Project Chimera Discussions](https://github.com/tomwolfe/project_chimera/discussions)
- **Twitter**: Follow [@Proj_Chimera](https://x.com/Proj_Chimera) for updates
- **Discord**: [Join our Discord server](https://discord.gg/projectchimera) (coming soon)

## 🙏 Acknowledgments

Project Chimera builds upon numerous open-source projects and frameworks. Special thanks to the Python, Streamlit, Bandit, Ruff, and GitHub Actions communities for their excellent tools and libraries.

---

> "The only true wisdom is in knowing you know nothing." - Socrates  
> Project Chimera embodies this philosophy through continuous self-reflection and improvement.