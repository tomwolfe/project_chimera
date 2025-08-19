# Project Chimera: Socratic Self-Debate Engine

[![GitHub](https://img.shields.io/badge/GitHub-Project_Chimera-000?style=flat-square&logo=github)](https://github.com/tomwolfe/project_chimera)
[![X (Twitter)](https://img.shields.io/badge/X-Proj_Chimera-1DA1F2?style=flat-square&logo=x)](https://x.com/Proj_Chimera)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Project%20Chimera-green?style=flat-square&logo=streamlit)](https://project-chimera-406972693661.us-central1.run.app)

*Note: Actual screenshot should be added here showing the interface in action*

## Overview

Project Chimera is an advanced reasoning engine that employs a Socratic debate methodology to solve complex problems and generate high-quality code. Unlike traditional single-model approaches, Chimera leverages multiple specialized "personas" that debate, critique, and refine solutions through a structured reasoning process, resulting in more robust and thoughtful outputs.

The system is designed with self-improvement capabilities, allowing it to analyze its own codebase and suggest meaningful enhancements based on the 80/20 Pareto principle.

## Key Features

- **Multi-Agent Socratic Debate**: Multiple specialized personas (Software Engineer, Test Engineer, DevOps Engineer, etc.) debate to refine solutions
- **Self-Improvement Capability**: Critically analyzes its own codebase to identify impactful improvements
- **Structured Reasoning Process**: Clear separation of analysis, critique, and synthesis phases
- **Robust Error Handling**: Comprehensive exception hierarchy and circuit breaker pattern for reliability
- **Security Analysis**: AST-based detection of specific vulnerabilities including:
  - `subprocess.run()` with `shell=True`
  - Unsafe `pickle.load()` usage
  - XML External Entity (XXE) vulnerabilities
- **Token Budget Management**: Prevents runaway costs with configurable token limits and fallback counting heuristics
- **Markdown Report Generation**: Detailed reasoning trace and final solution documentation
- **Docker Containerization**: Easy deployment with production-ready container configuration

## Architecture

Project Chimera follows a multi-layered architecture:

1. **Core Engine**: Manages the debate process, token budgeting, and persona coordination
2. **Persona System**: Specialized agents with domain expertise (Software Engineering, Testing, DevOps, etc.)
3. **Validation Layer**: Ensures output quality through schema validation and security checks
4. **Resilience Framework**: Circuit breakers and error handling for production stability
5. **Web Interface**: Streamlit-based UI for user interaction and visualization

## Getting Started

### Prerequisites

- Python 3.11+
- Google Gemini API Key (free tier available)
- Docker (for containerized deployment)

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/tomwolfe/project_chimera.git
cd project_chimera
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/MacOS
# venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your Gemini API key:
```bash
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

5. Run the application:
```bash
streamlit run app.py
```

### Docker Deployment

For production deployment:

```bash
docker build -t project-chimera .
docker run -p 8080:8080 -e GEMINI_API_KEY=your_api_key_here project-chimera
```

## Usage

1. Launch the application (either locally or via Docker)
2. Enter your Gemini API key in the sidebar
3. Select a task from the predefined categories:
   - Coding & Implementation
   - Testing & Quality Assurance
   - DevOps & Deployment
   - Self-Improvement Analysis
4. Submit your prompt or select an example
5. Watch the Socratic debate process unfold as multiple personas analyze, critique, and refine the solution
6. Review the final synthesized answer and detailed reasoning report

## Self-Improvement Capability

One of Chimera's most unique features is its ability to analyze its own codebase. When prompted with requests like "Critically analyze the entire Project Chimera codebase," the system will:

1. Conduct a thorough analysis of its own implementation
2. Identify high-impact improvement opportunities using the 80/20 principle
3. Propose specific, actionable code modifications
4. Provide clear rationales for each suggested change

This meta-cognitive capability enables continuous self-enhancement of the system's reasoning quality, robustness, and efficiency.

## Technical Highlights

- **Structured Error Handling**: Comprehensive exception hierarchy with domain-specific errors
- **Circuit Breaker Pattern**: Prevents cascading failures during LLM provider issues
- **Precise Security Analysis**: AST-based detection of specific security patterns in Python code
- **Adaptive Token Budgeting**: 
  - Uses Google's token counting API when available
  - Falls back to character-based heuristics (3.5 chars/token for code, 4 chars/token for text) when API fails
- **Production-Ready Logging**: Structured JSON logging for monitoring and debugging
- **Strict Schema Validation**: Ensures consistent output formats through JSON schema validation

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Create a new Pull Request

When contributing, please ensure:
- Your code follows the existing style conventions
- New features include appropriate test coverage
- Documentation is updated for significant changes
- You've run pre-commit hooks before submitting

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google for the Gemini API
- Streamlit for the web application framework
- The open-source community for various supporting libraries

## Contact

For questions, issues, or collaboration opportunities:

- Project Repository: [https://github.com/tomwolfe/project_chimera](https://github.com/tomwolfe/project_chimera)
- Twitter: [@Proj_Chimera](https://x.com/Proj_Chimera)
- Email: projectchimera@protonmail.com

---

*Project Chimera is an ongoing research project exploring advanced reasoning techniques through multi-agent debate systems. Note: The live demo requires JavaScript and may experience occasional availability issues.*