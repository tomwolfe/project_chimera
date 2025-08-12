# Project Chimera: Socratic Self-Debate Engine

[![GitHub](https://img.shields.io/badge/GitHub-Project_Chimera-000?style=flat-square&logo=github)](https://github.com/tomwolfe/project_chimera)
[![X (Twitter)](https://img.shields.io/badge/X-Proj_Chimera-1DA1F2?style=flat-square&logo=x)](https://x.com/Proj_Chimera)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Project%20Chimera-green?style=flat-square&logo=streamlit)](https://project-chimera-406972693661.us-central1.run.app)

## Overview

Project Chimera is an advanced reasoning engine that employs a Socratic debate methodology to solve complex problems through collaborative multi-agent discussions. By simulating a panel of specialized AI personas that debate and refine solutions through multiple critique cycles, Chimera achieves higher quality outputs than traditional single-model approaches.

Unlike conventional AI systems, Chimera implements a structured debate process where specialized personas sequentially analyze, critique, and refine solutions according to their domain expertise. The system culminates in a JSON-structured final output validated against strict schemas to ensure reliability.

## Key Features

- **Specialized Persona System**: 8+ dedicated personas with domain-specific expertise:
  - *Visionary_Generator*: Creates initial proposals with creativity
  - *DevOps_Specialist*: Focuses on CI/CD, infrastructure, and operational efficiency
  - *Security_Auditor*: Identifies vulnerabilities and enforces security best practices
  - *Test_Engineer*: Ensures test coverage and quality assurance
  - *Code_Refactorer*: Optimizes code structure and readability
  - *Devils_Advocate*: Challenges assumptions and identifies potential flaws
  - *Impartial_Arbitrator*: Synthesizes final JSON output (critical for system reliability)
  
- **Strict Output Validation**: 
  - Final outputs must adhere to precise JSON schemas
  - Code outputs validated against PEP8 (88 char line limits), Bandit security scans, and AST analysis
  - Automatic error recovery with fallback mechanisms when validation fails

- **Context-Aware Analysis**:
  - Semantic embedding-based context relevance scoring (using SentenceTransformer)
  - Dynamic codebase context weighting based on keyword analysis
  - Negation-aware proximity scoring for more accurate context selection

- **Robust Production Architecture**:
  - Circuit breakers for API reliability (configurable failure thresholds)
  - Rate limiting to prevent quota exhaustion
  - Token budget management with configurable context allocation ratios
  - Comprehensive error handling with detailed logging

- **Self-Improvement Capability**:
  - Can analyze its own codebase to identify improvements
  - Generates specific code modification suggestions with context-aware diffs
  - Produces actionable technical debt reports

- **Comprehensive Reporting**:
  - Detailed markdown reports with complete reasoning history
  - Token usage tracking by persona and phase
  - Persona configuration audit trails
  - Malformed block identification for troubleshooting

## Architecture

Project Chimera employs a sophisticated multi-agent architecture with strict process control:

1. **Persona Router**: Selects appropriate personas based on problem domain and prompt analysis
   - Uses keyword matching with negation awareness
   - Supports domain-specific persona sets (Software Engineering, Strategic Analysis, Creative)
   - Dynamic persona sequencing based on debate progress

2. **Debate Orchestrator**: Manages the complete Socratic debate process
   - Enforces strict phase sequencing (proposal → critique → synthesis)
   - Tracks intermediate reasoning steps with token accounting
   - Implements circuit breakers for each persona generation phase

3. **Context Relevance Analyzer**: 
   - Computes semantic embeddings for code context files
   - Applies keyword-based boosting with negation handling
   - Returns context-weighted file relevance scores for optimal context selection

4. **Validation Pipeline**:
   - Schema validation using Pydantic models
   - Code-specific validation (PEP8, security, syntax)
   - Automatic recovery from malformed outputs
   - Malformed block identification and reporting

5. **Resilience Layer**:
   - Rate limiting with configurable thresholds
   - Circuit breakers with configurable failure thresholds and recovery timeouts
   - Token budget enforcement with configurable context allocation ratios
   - Comprehensive error logging with structured context

## Getting Started

### Prerequisites

- Python 3.10+
- Google API key for Gemini access (with `gemini-2.5-pro` or `gemini-2.5-flash` permissions)
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
```

### Configuration

1. Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your_google_api_key_here
   ```

2. For development, create a `secrets.toml` file in the `.streamlit` directory:
   ```toml
   [server]
   enableCORS = false
   enableXsrfProtection = true
   
   [gemini]
   api_key = "your_google_api_key_here"
   ```

### Running the Application

```bash
streamlit run app.py
```

The application will launch in your default web browser. For production deployments, see the Docker section below.

## Usage

### Web Interface Features

The Streamlit interface provides:

- **Prompt Templates**: Pre-configured prompts organized by category (Software Engineering, Strategic Analysis, Creative)
- **Framework Selection**: Choose between specialized reasoning frameworks
- **Context Management**: Upload codebases for context-aware analysis
- **Resource Controls**: Configure token budgets and context allocation ratios
- **Detailed Process Logging**: Real-time visibility into the debate process
- **Exportable Reports**: Download comprehensive markdown reports with complete reasoning history

### Self-Analysis Capability

Use the "Critically analyze the entire Project Chimera codebase" prompt to generate actionable improvement suggestions with specific code modifications. The system will:
1. Analyze the codebase structure
2. Identify security concerns and maintainability issues
3. Generate specific code change proposals with context
4. Provide a detailed technical debt assessment

### Output Format Requirements

The Impartial_Arbitrator persona enforces strict JSON output format:
```json
{
  "COMMIT_MESSAGE": "Clear summary of changes",
  "RATIONALE": "Detailed justification for changes",
  "CODE_CHANGES": [
    {
      "FILE_PATH": "path/to/file.py",
      "ACTION": "ADD|MODIFY|REMOVE",
      "FULL_CONTENT": "Complete file content (for ADD/MODIFY)",
      "LINES": ["Specific lines to remove (for REMOVE)"]
    }
  ],
  "CONFLICT_RESOLUTION": "How conflicts were resolved",
  "UNRESOLVED_CONFLICT": "Any remaining conflicts",
  "malformed_blocks": []
}
```

## Docker Deployment

For production deployment:

```bash
# Build the Docker image
docker build -t project-chimera .

# Run the container
docker run -p 8080:8080 -e GEMINI_API_KEY=your_api_key_here project-chimera
```

The application will be available at `http://localhost:8080`.

## Project Structure

```
project_chimera/
├── src/
│   ├── config/             # Configuration management
│   ├── context/            # Context analysis and relevance scoring
│   ├── exceptions/         # Custom exception hierarchy
│   ├── middleware/         # Rate limiting and request processing
│   ├── models/             # Pydantic validation schemas
│   ├── persona/            # Persona management and routing
│   ├── resilience/         # Circuit breakers and error handling
│   ├── tokenizers/         # Token counting implementations
│   ├── utils/              # Utility functions and helpers
│   └── core.py             # SocraticDebate engine implementation
├── app.py                  # Streamlit web interface
├── personas.yaml           # Persona configurations and system prompts
├── requirements.txt        # Python dependencies
├── .dockerignore
├── .gitignore
├── .pre-commit-config.yaml # Pre-commit hooks configuration
└── Dockerfile
```

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a new feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Create a new Pull Request

Please ensure your code follows the project's pre-commit hooks (configured in `.pre-commit-config.yaml`), which include:
- Black code formatting
- Ruff linting and fixing
- AST syntax checking
- YAML validation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions or issues, please [open an issue](https://github.com/tomwolfe/project_chimera/issues) on GitHub.

---

*Project Chimera: Where AI debates itself to find better answers.*