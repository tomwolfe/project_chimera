# Project Chimera: Socratic Self-Debate Engine

![Project Status](https://img.shields.io/badge/status-active-green)
![License](https://img.shields.io/badge/license-MIT-blue)

An advanced reasoning engine that employs Socratic debate methodology between specialized AI personas to solve complex problems and generate high-quality code solutions. Project Chimera doesn't just provide answers—it debates them to ensure robustness, correctness, and optimal implementation through transparent, structured reasoning with comprehensive audit trails.

## Overview

Project Chimera implements a sophisticated self-debate framework where specialized AI personas critically analyze problems from multiple perspectives. Unlike traditional LLMs that generate single responses, Chimera engages in a structured debate process with:

- **Dynamic Token Allocation**: Intelligent distribution of token budget across context analysis (10-30%), debate (60-80%), and synthesis (5-15%) phases based on prompt complexity, with minimum thresholds (e.g., `max(400, int(available_tokens * synthesis_ratio))`)
- **Semantic Context Analysis**: Uses SentenceTransformer (`all-MiniLM-L6-v2`) with cosine similarity to identify relevant code files through semantic embeddings
- **Multi-Persona Debate**: Specialized agents with defined roles, temperature settings, token limits, and dynamic sequencing
- **Structured Validation**: Pydantic-based schema validation with multi-attempt recovery and detailed error reporting
- **Comprehensive Audit Trail**: Complete tracking of persona parameter adjustments, intermediate steps, and process logs
- **Domain-Aware Framework Selection**: Automatic framework detection using weighted keyword matching with negation handling

## Key Features

### 🧠 Socratic Self-Debate Framework
- **Adaptive Token Budgeting**:
  - Context ratio dynamically adjusts between 10-30% based on prompt complexity
  - Minimum token thresholds ensure critical phases always have sufficient resources
  - Real-time token usage monitoring with budget enforcement via `TokenBudgetExceededError`
  - Detailed cost tracking with `Total_Estimated_Cost_USD` in intermediate steps
- **Multi-Phase Reasoning**:
  - *Context Analysis*: Semantic codebase understanding using vector embeddings
  - *Structured Debate*: Multiple specialized personas critique and improve proposals
  - *Conflict Resolution*: Impartial Arbitrator synthesizes consensus from conflicting viewpoints
- **Dynamic Persona Sequencing**:
  - Base sequence adjusted based on prompt content and intermediate results
  - Quality metrics trigger persona additions (e.g., adding `Devils_Advocate` if `reasoning_depth < 0.6`)
  - Negation handling prevents misclassification (e.g., "not secure" doesn't trigger security analysis)
  - Self-analysis prompts use specialized sequence: `["Context_Aware_Assistant", "Code_Architect", "Security_Auditor", "Constructive_Critic", "Test_Engineer", "DevOps_Engineer", "Impartial_Arbitrator"]`
- **Comprehensive Audit Trail**:
  - Full logging of persona parameter adjustments with before/after values
  - Intermediate steps tracking with input/output token counts per persona
  - Process logs with ANSI codes stripped for clean viewing
  - Detailed error reporting with raw output snippets

### 💻 Software Engineering Focus
- **Context-Aware Code Analysis**:
  - Semantic code relevance scoring with weighted keyword matching (e.g., +0.1 boost for path matches)
  - Context ratio adapts based on prompt complexity: `context_ratio = max(0.1, min(0.3, base_ratio + complexity_score * 0.05))`
  - File-specific analysis with line-number precision
  - Quality metrics including:
    - `code_quality_score` (0.0-1.0)
    - `complexity_score`
    - `maintainability_index`
    - `test_coverage_estimate`
- **Specialized Personas**:
  - *Code_Architect*: Structural integrity, scalability (temp: 0.3, tokens: 1024)
  - *Constructive_Critic*: Logical gaps, security vulnerabilities (temp: 0.45, tokens: 2048)
  - *Security_Specialist*: Input validation, authentication (temp: 0.3, tokens: 2048)
  - *Testing_Strategist*: Test coverage strategies (temp: 0.3, tokens: 2048)
  - *Context_Aware_Assistant*: Codebase-specific analysis with quality metrics (temp: 0.1, tokens: 1024)
  - *Devils_Advocate*: Challenges proposed improvements (temp: 0.6, tokens: 2048)
  - *Impartial_Arbitrator*: Synthesizes final solution with conflict resolution (temp: 0.2, tokens: 4096)
- **Structured Output Format**:
  ```json
  {
    "COMMIT_MESSAGE": "Refactor context analyzer for improved token efficiency",
    "RATIONALE": "CONFLICT_RESOLUTION: After debate, the team determined...",
    "CONFLICT_RESOLUTION": "Resolved X vs Y by prioritizing security over performance",
    "UNRESOLVED_CONFLICT": null,
    "CODE_CHANGES": [
      {
        "FILE_PATH": "src/context_analyzer.py",
        "ACTION": "MODIFY",
        "FULL_CONTENT": "import numpy as np\n# Improved vectorization..."
      }
    ],
    "quality_metrics": {
      "code_quality_score": 0.85,
      "complexity_score": 0.65,
      "maintainability_index": 0.82,
      "test_coverage_estimate": 0.65
    }
  }
  ```

### ⚙️ Advanced Technical Implementation
- **Semantic Context Analysis**:
  - Uses `SentenceTransformer("all-MiniLM-L6-v2")` with cosine similarity for code relevance scoring
  - Implements weighted similarity with keyword boosting (e.g., +0.2 for test files)
  - Context ratio adapts based on prompt complexity: `context_ratio = max(0.1, min(0.3, base_ratio + complexity_score * 0.05))`
- **Validation System**:
  - Pydantic models ensure schema compliance with multi-attempt processing
  - Detailed error reporting with malformed block classification:
    ```json
    {
      "type": "SCHEMA_VALIDATION_ERROR",
      "message": "Field required",
      "raw_string_snippet": "{\"COMMIT_MESSAGE\": \"Fix bug\"}"
    }
    ```
  - Automatic JSON extraction from markdown code blocks
  - Fallback mechanisms for partial data recovery
- **Domain Detection**:
  - Weighted keyword-based framework selection with negation handling
  - Self-analysis detection uses specific weighted phrases (e.g., "analyze the entire Project Chimera codebase": 1.0)
  - Prevents misclassification using proximity-based negation analysis
- **Error Handling**:
  - Comprehensive exception hierarchy:
    - `TokenBudgetExceededError`: Raised when token usage exceeds budget
    - `SchemaValidationError`: Output fails schema validation
    - `LLMResponseValidationError`: Response cannot be parsed
    - `ChimeraError`: Base class for all custom exceptions
  - Detailed error reporting with raw output snippets for debugging

### 🖥️ Comprehensive Reporting
- **Detailed Markdown Reports** including:
  - Persona configuration audit trail showing parameter changes
  - Process log with ANSI codes stripped for clean viewing
  - Intermediate reasoning steps with token usage metrics
  - Final synthesized answer with complete schema validation
  - Quality metrics and cost analysis
- **Multiple Output Formats**:
  - Human-readable markdown reports with date stamps
  - Machine-parsable JSON for integration
  - Downloadable structured outputs with full context

## Installation

```bash
# Clone the repository
git clone https://github.com/tomwolfe/project_chimera.git
cd project_chimera

# Create and activate virtual environment (Python 3.9+ recommended)
python -m venv venv
source venv/bin/activate  # Linux/MacOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

## Usage

### Web Interface (Recommended)
```bash
streamlit run app.py
```

The web application provides:
- Prompt input with example templates and automatic framework recommendation
- Codebase context upload (up to 100 files with various extensions)
- Configurable token budgets (5,000-128,000 tokens) and context ratio (5%-50%)
- Real-time persona configuration editing with reset-to-default functionality
- Visualized debate process with complete intermediate steps
- Downloadable markdown reports and JSON outputs
- Persona audit trail showing configuration changes
- Process log with ANSI codes stripped for clean viewing

### Basic CLI Usage
```python
from core import ChimeraEngine

engine = ChimeraEngine(
    api_key="your_gemini_key",
    context_token_budget_ratio=0.25,  # 25% of tokens for context analysis
    max_total_tokens_budget=8000
)

response = engine.run(
    user_prompt="Implement a Python function to calculate Fibonacci sequence",
    codebase_context={"fib.py": "def fib(n):..."},  # Optional code context
    domain="Software Engineering"
)

print(response)
```

## Reasoning Frameworks

### Software Engineering Framework (Default)
Specialized for code generation, analysis, and improvement with:

| Persona | Focus Area | Temperature | Max Tokens | Key Responsibilities |
|---------|------------|-------------|------------|----------------------|
| Code_Architect | Structural integrity | 0.3 | 1024 | Modularity, scalability, maintainability |
| Constructive_Critic | Improvement suggestions | 0.45 | 2048 | Logical gaps, security vulnerabilities |
| Security_Specialist | Security analysis | 0.3 | 2048 | Input validation, authentication, authorization |
| Testing_Strategist | Test coverage | 0.3 | 2048 | Comprehensive test strategies |
| Context_Aware_Assistant | Code context | 0.1 | 1024 | Quality metrics, file-specific analysis |
| Devils_Advocate | Critical challenge | 0.6 | 2048 | Identifies over-correction, hidden assumptions |
| Impartial_Arbitrator | Synthesis | 0.2 | 4096 | Conflict resolution, final decision |

### Self-Analysis Framework
Triggered automatically for Project Chimera self-improvement with:

| Persona | Focus Area | Temperature | Max Tokens | Key Responsibilities |
|---------|------------|-------------|------------|----------------------|
| Context_Aware_Assistant | Codebase context | 0.1 | 1024 | Quality metrics, file-specific analysis |
| Code_Architect | Architecture | 0.3 | 1024 | Technical debt, separation of concerns |
| Security_Auditor | Security | 0.3 | 2048 | Vulnerability analysis |
| Constructive_Critic | Improvement | 0.45 | 2048 | Best practices, code quality |
| Test_Engineer | Testing | 0.3 | 2048 | Test coverage strategies |
| DevOps_Engineer | Deployment | 0.3 | 2048 | CI/CD, infrastructure, monitoring |
| Impartial_Arbitrator | Synthesis | 0.2 | 4096 | Conflict resolution, final decision |

### General Reasoning Framework
For non-technical problem solving with:
- Critical_Thinker
- Creative_Innovator
- Detail_Oriented_Analyst
- Pragmatic_Implementer

## Self-Improvement Capabilities

Project Chimera can analyze and improve its own codebase using structured quality metrics:

```text
"Critically analyze the entire Project Chimera codebase. Identify the most impactful code changes for self-improvement, focusing on the 80/20 Pareto principle. Prioritize enhancements to reasoning quality, robustness, efficiency, and developer maintainability. For each suggestion, provide a clear rationale and a specific, actionable code modification."
```

The system generates detailed analysis including:
- `key_modules` with responsibility, quality scores, and criticality
- `architectural_pattern` identification
- `quality_metrics` with numerical scores (0.0-1.0)
- `critical_concerns` with specific file:line references
- Automatic test file generation for code changes

## Output Validation

All outputs undergo rigorous multi-stage validation:
1. **Schema Validation**: Pydantic models ensure required fields are present
2. **Malformed Block Detection**: Identifies and categorizes problematic sections:
   ```json
   {
     "type": "UI_PARSING_ERROR",
     "message": "Final answer was not a dictionary or list",
     "raw_string_snippet": "{\"COMMIT_MESSAGE\": \"Fix bug\"}"
   }
   ```
3. **Multi-Attempt Processing**: Up to 2 retries for JSON formatting/schema issues
4. **Fallback Mechanisms**: Partial data recovery for critical fields
5. **Detailed Error Reporting**: Includes raw output snippets for debugging

Example validation error handling:
```json
{
  "COMMIT_MESSAGE": "Debate Failed - Schema Validation",
  "RATIONALE": "A schema validation error occurred",
  "CODE_CHANGES": [],
  "CONFLICT_RESOLUTION": null,
  "UNRESOLVED_CONFLICT": null,
  "error_details": {
    "type": "SCHEMA_VALIDATION_ERROR",
    "message": "Field required",
    "raw_string_snippet": "{\"COMMIT_MESSAGE\": \"Fix bug\"}"
  }
}
```

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a new feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Create a new Pull Request

When contributing code improvements, please use Project Chimera itself to generate your proposed changes:
1. Load the codebase in the web interface
2. Use the "Critically analyze the entire Project Chimera codebase..." prompt
3. Review the generated suggestions and rationale
4. Implement the most impactful changes with proper justification

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Project Chimera builds upon the principles of Socratic dialogue and leverages Google's Gemini API for its language model capabilities. The project incorporates advanced NLP techniques including semantic embeddings with SentenceTransformer and structured output validation with Pydantic.

---

*Project Chimera: Where AI doesn't just answer questions—it debates them to find the best possible solutions, with transparent reasoning you can follow step by step.*