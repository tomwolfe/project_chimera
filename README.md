# Project Chimera: Socratic Self-Debate Reasoning Engine

An advanced reasoning engine for complex problem-solving and code generation that leverages Socratic debate methodology to enhance AI reasoning quality through structured self-critique and refinement.

## 🌟 Overview

Project Chimera is an innovative framework that enables AI systems to engage in Socratic self-debate, where multiple specialized personas critically analyze problems, challenge assumptions, and collaboratively arrive at higher-quality solutions. Rather than relying on a single response, Chimera orchestrates a debate process where different perspectives examine the problem from multiple angles before synthesizing a final, more robust answer.

The system features a sophisticated self-improvement engine that can analyze its own codebase to identify high-impact improvements across four critical dimensions: **Reasoning Quality**, **Robustness**, **Efficiency**, and **Maintainability** - applying the 80/20 Pareto principle to prioritize the most impactful changes.

## ✨ Key Features

-   **Socratic Debate Framework**: Multiple AI personas engage in structured debate to refine solutions
-   **Self-Improvement Engine**: Critically analyze the entire Project Chimera codebase to identify high-impact improvements with specific code modification suggestions
-   **Structured Output Format**: Consistent JSON responses with detailed improvement suggestions:
    ```json
    "IMPACTFUL_SUGGESTIONS": [{
      "AREA": "Reasoning Quality|Robustness|Efficiency|Maintainability",
      "PROBLEM": "<string>",
      "PROPOSED_SOLUTION": "<string>",
      "EXPECTED_IMPACT": "<string>",
      "CODE_CHANGES_SUGGESTED": [{
        "FILE_PATH": "<string>",
        "ACTION": "ADD|MODIFY|REMOVE",
        "FULL_CONTENT": "<string>",
        "LINES": ["<string>", "<string>"]
      }]
    }],
    "malformed_blocks": []
    ```
-   **Dynamic Persona Sequencing**: Context-aware selection and ordering of specialized personas (Code Architect, Security Auditor, Test Engineer, DevOps Engineer, etc.)
-   **Real-time Metrics**: Monitor token usage, reasoning quality, and debate progress with objective metrics collection
-   **Codebase Analysis**: Upload your code for context-aware analysis and improvement suggestions
-   **Error Resilience**: Comprehensive error handling with circuit breaker pattern and adaptive retry mechanisms
-   **Multi-framework Support**: Specialized reasoning for software engineering, security, architecture, and more

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Gemini API key (get one from [Google AI Studio](https://aistudio.google.com/apikey))

### Installation
```bash
# Clone the repository
git clone https://github.com/tomwolfe/project_chimera.git
cd project_chimera

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/MacOS
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration
1. Create a `.env` file in the project root
2. Add your Gemini API key:
    ```
    GEMINI_API_KEY=your_api_key_here
    ```

### Running the Application
```bash
streamlit run app.py
```

## 💻 Usage

1. Launch the web interface via Streamlit.
2. Enter your Gemini API key in the sidebar.
3. Select a reasoning framework (e.g., "Software Engineering" for code-related tasks).
4. Choose from example prompts or create your custom prompt.
5. For code analysis tasks:
    - Select the "Software Engineering" framework.
    - Upload relevant code files.
    - Enter your prompt about the codebase.
6. Click "Run" to start the Socratic debate process.
7. View the debate history, metrics, and final synthesized answer.

![Project Chimera Interface](https://github.com/tomwolfe/project_chimera/raw/main/docs/interface.png)

## 🛠️ Self-Improvement Analysis

One of Chimera's most powerful features is its ability to analyze itself with surgical precision. By selecting the "Self-Improvement" framework and a prompt like "Critically analyze the entire Project Chimera codebase," the system performs a deep self-evaluation.

The output is structured as a `SelfImprovementAnalysisOutput` JSON object, providing actionable insights and concrete code modification suggestions.

### How it Works:

1.  **Select Framework & Prompt:** Choose the "Self-Improvement" framework and a self-analysis prompt.
2.  **Objective Metrics Collection:** The system gathers data on its own performance, including:
    *   **Code Quality:** PEP8 compliance, cyclomatic complexity, lines of code per function, argument count, nesting depth, code smells, and potential bottlenecks.
    *   **Security:** Bandit findings and AST-based security vulnerability checks.
    *   **Performance & Efficiency:** Token usage per phase, total cost, debate efficiency (turns, malformed blocks, conflict resolution), and average turn tokens.
    *   **Robustness:** Schema validation failures, presence of unresolved conflicts, and conflict resolution attempts.
    *   **Maintainability:** Basic assessment of test coverage (though actual coverage percentage requires running tests).
3.  **Persona Debate:** Specialized personas (like `Self_Improvement_Analyst`, `Code_Architect`, `Security_Auditor`, `DevOps_Engineer`, `Test_Engineer`, `Constructive_Critic`, `Impartial_Arbitrator`, `Devils_Advocate`) debate the findings and potential improvements.
4.  **Structured Output:** The final analysis is synthesized into a `SelfImprovementAnalysisOutput` JSON object.

### `SelfImprovementAnalysisOutput` Structure:

The output is designed to be comprehensive and actionable, detailing findings across four key areas:

```json
{
  "ANALYSIS_SUMMARY": "A high-level summary of the overall self-improvement findings and priorities.",
  "IMPACTFUL_SUGGESTIONS": [
    {
      "AREA": "Reasoning Quality | Robustness | Efficiency | Maintainability",
      "PROBLEM": "A clear description of the identified issue or area for improvement.",
      "PROPOSED_SOLUTION": "A specific suggestion for how to address the problem.",
      "EXPECTED_IMPACT": "An explanation of the anticipated benefits of implementing the solution (e.g., 'Reduces token usage by 15%', 'Improves readability', 'Mitigates security risk').",
      "CODE_CHANGES_SUGGESTED": [
        {
          "FILE_PATH": "Path/to/the/relevant/file.py",
          "ACTION": "ADD | MODIFY | REMOVE",
          "FULL_CONTENT": "The complete new content for the file (for ADD/MODIFY actions).",
          "LINES": ["Line of code to remove", "Another line to remove"]
        }
      ]
    }
    // ... more suggestions
  ],
  "malformed_blocks": [
    {
      "type": "Error Type (e.g., JSON_EXTRACTION_FAILED, SCHEMA_VALIDATION_ERROR)",
      "message": "Details about the malformation or error.",
      "raw_string_snippet": "Snippet of the problematic output."
    }
  ]
}
```

This detailed output allows users (or the AI itself) to understand the rationale behind suggested changes and directly apply them to improve the codebase.

## 📂 Project Structure

```
project_chimera/
├── src/                    # Core application code
│   ├── core/               # Debate engine and core logic
│   │   ├── debate_engine.py  # Main debate orchestration
│   │   ├── persona_manager.py # Dynamic persona sequencing
│   ├── personas/           # Persona definitions and configurations
│   ├── utils/              # Utility functions and helpers
│   ├── self_improvement/   # Self-analysis components
│   └── exceptions.py       # Custom exception hierarchy
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── .dockerignore           # Docker ignore configuration
├── LICENSE                 # MIT License
└── README.md               # This file
```

## 🧪 Technical Highlights

### Dynamic Persona System
- Context-aware persona selection based on prompt analysis
- Specialized personas for different domains (Code Architect, Security Auditor, Test Engineer, DevOps Engineer)
- Dynamic sequencing that adapts based on detected keywords and analysis context
- Performance-based adaptive tuning of persona parameters (temperature, max tokens)

### Self-Improvement Engine
- Comprehensive metrics collection for objective self-evaluation
- AST-based code analysis for detailed code metrics
- Structured JSON output with specific code modification suggestions
- 80/20 Pareto principle applied to prioritize high-impact improvements
- Real-time validation of code suggestions for syntax and style compliance

### Error Resilience
- Circuit breaker pattern implementation for API failures
- Adaptive retry mechanisms with exponential backoff
- Rate limit monitoring and handling
- Comprehensive error logging and recovery strategies

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1.  Fork the repository
2.  Create a new feature branch (`git checkout -b feature/your-feature`)
3.  Commit your changes (`git commit -am 'Add some feature'`)
4.  Push to the branch (`git push origin feature/your-feature`)
5.  Create a new Pull Request

Please ensure your code follows our style guidelines and includes appropriate tests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📬 Contact

*   **GitHub**: [tomwolfe/project_chimera](https://github.com/tomwolfe/project_chimera)
*   **Twitter**: [@Proj_Chimera](https://x.com/Proj_Chimera  )
*   **Email**: projectchimera.dev@gmail.com

---

*Project Chimera is actively developed and maintained by the open-source community. Your contributions and feedback are invaluable to our mission of creating more robust, transparent, and reliable AI reasoning systems.*