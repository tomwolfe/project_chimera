# Project Chimera: Socratic Self-Debate Engine

Project Chimera is an advanced reasoning engine that leverages Socratic debate methodology to produce higher-quality AI responses through multi-perspective analysis. By simulating a debate between specialized AI personas, it identifies flaws, enhances reasoning quality, and delivers more robust solutions to complex problems. Unique among AI reasoning frameworks, Chimera features a **self-improvement capability** that allows it to critically analyze and enhance its own codebase.

*   **GitHub**: [tomwolfe/project_chimera](https://github.com/tomwolfe/project_chimera)
*   **Twitter**: [@Proj_Chimera](https://x.com/Proj_Chimera)

## Key Features

- **Socratic Self-Debate Framework**: Multiple specialized AI personas debate to refine answers and identify weaknesses through structured multi-turn reasoning
- **Self-Improvement Capabilities**: The system can critically analyze its own codebase to identify areas for enhancement using the 80/20 Pareto principle
- **Domain-Specific Reasoning**: Specialized personas for different domains (Software Engineering, Scientific, Business, Security, DevOps, etc.)
- **Structured Output**: Strict JSON validation with automatic repair of malformed outputs and detailed diagnostics
- **Comprehensive Code Validation**: Integrated Ruff linting, Bandit security scanning, and custom AST-based security analysis
- **Token Budget Management**: Intelligent allocation of token resources across debate phases with cost tracking
- **Markdown Report Generation**: Detailed analysis reports with persona audit trails and intermediate reasoning steps
- **Context-Aware File Selection**: Intelligent identification of relevant code files for modification based on prompt analysis

## Technical Overview

Project Chimera is built with modern Python technologies and follows rigorous software engineering practices:

- **Core Framework**: Python 3.11+
- **UI**: Streamlit for interactive web interface with responsive design
- **LLM Integration**: Gemini 2.5 series (primary), with extensible architecture for other providers
- **Validation Tools**: 
  - Ruff for code style and quality
  - Bandit for security scanning
  - Custom AST-based security analysis
  - JSON output validation with repair capabilities
- **Code Quality**: Strict schema validation using Pydantic models
- **Containerization**: Docker support for consistent deployment
- **Structured Logging**: Comprehensive monitoring with rich console output
- **Test Coverage**: Unit tests for output parsing, validation, and core functionality

## Installation

### Prerequisites
- Python 3.11+
- Google Cloud account (for Gemini API access)
- Docker (optional, for containerized deployment)

### Local Setup
```bash
# Clone the repository
git clone https://github.com/tomwolfe/project_chimera.git
cd project_chimera

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/MacOS
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set up API keys
echo "GEMINI_API_KEY=your_api_key_here" > .env

# Run the application
streamlit run app.py
```

### Docker Setup
```bash
# Build the container
docker build -t project-chimera .

# Run the container
docker run -p 8080:8080 -e GEMINI_API_KEY=your_api_key_here project-chimera
```

## Usage

### Basic Interaction
1. Launch the application via Streamlit
2. Enter your prompt in the main interface
3. Select the appropriate reasoning domain (Software Engineering, Scientific, Business, etc.)
4. Configure parameters as needed (token budget, model selection)
5. Initiate the Socratic debate process
6. Review the structured output with debate history and final synthesis

### Self-Improvement Analysis
For analyzing and improving the Project Chimera codebase itself, use prompts like:
```
Critically analyze the entire Project Chimera codebase. Identify the most impactful code changes for self-improvement, focusing on the 80/20 Pareto principle. Prioritize enhancements to reasoning quality, robustness, efficiency, and developer maintainability.
```

The system will:
1. Analyze code quality metrics (Ruff violations, complexity metrics)
2. Identify security vulnerabilities (Bandit scans, AST analysis)
3. Evaluate architectural patterns and potential bottlenecks
4. Generate specific, actionable code change suggestions
5. Provide detailed rationale and expected impact for each suggestion

### Report Generation
After any analysis, you can download comprehensive reports in Markdown format containing:
- Full debate history with token usage metrics
- Persona configuration audit trail
- Intermediate reasoning steps
- Final synthesized answer
- Code change suggestions with context
- Security and quality metrics

## Architecture Deep Dive

### Core Components

#### 1. Persona System
Specialized AI roles with distinct perspectives and system prompts:
- **Visionary Generator**: Proposes innovative solutions
- **Constructive Critic**: Identifies flaws and risks
- **Skeptical Generator**: Challenges assumptions
- **Domain-specific Experts**:
  - *Security_Auditor*: Focuses on vulnerabilities and compliance
  - *DevOps_Engineer*: Optimizes operational efficiency and deployment
  - *Test_Engineer*: Identifies testing gaps and quality issues
  - *Business_Analyst*: Evaluates business impact and value
  - *Self_Improvement_Analyst*: Specifically designed for codebase self-analysis

#### 2. Debate Engine
Orchestrates the multi-turn debate process with:
- Dynamic persona sequencing based on prompt analysis
- Token budget allocation across debate phases
- Conflict resolution mechanisms
- Intermediate step validation

#### 3. Output Validation Framework
Ensures high-quality, structured responses through:
- Strict JSON schema validation using Pydantic models
- Automatic repair of malformed JSON outputs
- Malformed block detection and reporting
- Code change validation (PEP8 compliance, path validation)

#### 4. Code Analysis System
Comprehensive code quality and security assessment:
- Ruff linting for style and quality
- Bandit security scanning
- Custom AST-based security analysis (detects unsafe patterns like `pickle.loads()`, `yaml.load()`)
- Complexity metrics collection (cyclomatic complexity, LOC, nesting depth)
- Code smell detection

#### 5. Self-Improvement Framework
Analyze the system's own code for enhancements:
- Automated metrics collection (code quality, security, performance)
- 80/20 prioritization of maintenance tasks
- Structured output format for improvement suggestions
- Code change recommendations with specific file paths

### Self-Analysis Output Structure
When analyzing its own codebase, Chimera produces structured output with:
```json
{
  "ANALYSIS_SUMMARY": "High-level assessment of codebase health",
  "IMPACTFUL_SUGGESTIONS": [
    {
      "AREA": "Reasoning Quality | Robustness | Efficiency | Maintainability",
      "PROBLEM": "Specific issue identified",
      "PROPOSED_SOLUTION": "Clear implementation guidance",
      "EXPECTED_IMPACT": "Quantified or qualitative benefit",
      "CODE_CHANGES_SUGGESTED": [
        {
          "FILE_PATH": "relative/path/to/file.py",
          "ACTION": "ADD | MODIFY | REMOVE",
          "START_LINE": 42,
          "END_LINE": 45,
          "CONTENT": "Valid code snippet"
        }
      ]
    }
  ]
}
```

## Security Features

Project Chimera incorporates multiple layers of security analysis:
- **AST-based Security Checks**: Detects dangerous patterns like:
  - `pickle.loads()` with untrusted data
  - `yaml.load()` without safe loader
  - Shell injection vulnerabilities
- **Bandit Integration**: Automated security scanning for common vulnerabilities
- **Context-Aware Analysis**: Security focus intensifies when analyzing security-related code
- **Self-Audit Capability**: Can identify and suggest fixes for its own security vulnerabilities

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a new feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Create a new Pull Request

### Contribution Guidelines
- All code must pass Ruff linting
- Security-sensitive code requires AST security analysis
- New features should include appropriate persona configurations
- Self-improvement suggestions must follow the structured output format
- All output parsers require corresponding test cases

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Project Chimera builds upon the rich tradition of Socratic dialogue and leverages modern AI capabilities to enhance reasoning quality. Special thanks to the open-source community for the foundational technologies that make this project possible.

---

*Project Chimera is actively developed and maintained. Check back frequently for new features and improvements!*