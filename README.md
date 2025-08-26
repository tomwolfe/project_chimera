# Project Chimera: Socratic Self-Debate Reasoning Engine

![Project Chimera Banner](https://github.com/tomwolfe/project_chimera/raw/main/docs/banner.png)

An advanced reasoning engine for complex problem-solving and code generation that leverages Socratic debate methodology to enhance AI reasoning quality through structured self-critique and refinement.

## 🌟 Overview

Project Chimera is an innovative framework that enables AI systems to engage in Socratic self-debate, where multiple specialized personas critically analyze problems, challenge assumptions, and collaboratively arrive at higher-quality solutions. Rather than relying on a single response, Chimera orchestrates a debate process where different perspectives examine the problem from multiple angles before synthesizing a final, more robust answer.

The system features a sophisticated self-improvement engine that can analyze its own codebase to identify high-impact improvements across four critical dimensions: **Reasoning Quality**, **Robustness**, **Efficiency**, and **Maintainability** - applying the 80/20 Pareto principle to prioritize the most impactful changes.

## ✨ Key Features

- **Socratic Debate Framework**: Multiple AI personas engage in structured debate to refine solutions
- **Self-Improvement Engine**: Critically analyze the entire Project Chimera codebase to identify high-impact improvements with specific code modification suggestions
- **Structured Output Format**: Consistent JSON responses with detailed improvement suggestions:
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
- **Dynamic Persona Sequencing**: Context-aware selection and ordering of specialized personas (Code Architect, Security Auditor, Test Engineer, DevOps Engineer, etc.)
- **Real-time Metrics**: Monitor token usage, reasoning quality, and debate progress with objective metrics collection
- **Codebase Analysis**: Upload your code for context-aware analysis and improvement suggestions
- **Error Resilience**: Comprehensive error handling with circuit breaker pattern and adaptive retry mechanisms
- **Multi-framework Support**: Specialized reasoning for software engineering, security, architecture, and more

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

1. Launch the web interface via Streamlit
2. Enter your Gemini API key in the sidebar
3. Select a reasoning framework (e.g., "Software Engineering" for code-related tasks)
4. Choose from example prompts or create your custom prompt
5. For code analysis tasks:
   - Select "Software Engineering" framework
   - Upload relevant code files
   - Enter your prompt about the codebase
6. Click "Run" to start the Socratic debate process
7. View the debate history, metrics, and final synthesized answer

![Project Chimera Interface](https://github.com/tomwolfe/project_chimera/raw/main/docs/interface.png)

## 🛠️ Self-Improvement Analysis

One of Chimera's most powerful features is its ability to analyze itself with surgical precision:

1. Select the "Self-Improvement" framework
2. Choose the "Critically analyze the entire Project Chimera codebase" prompt
3. The system will:
   - Evaluate its own reasoning quality, robustness, and efficiency using objective metrics
   - Identify high-impact improvement opportunities (applying the 80/20 Pareto principle)
   - Provide specific, actionable code modification suggestions with file paths and line numbers
   - Prioritize enhancements to reasoning quality, robustness, efficiency, and maintainability
   - Analyze code quality metrics including cyclomatic complexity, LOC, nesting depth, and code smells

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

1. Fork the repository
2. Create a new feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Create a new Pull Request

Please ensure your code follows our style guidelines and includes appropriate tests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📬 Contact

*   **GitHub**: [tomwolfe/project_chimera](https://github.com/tomwolfe/project_chimera)
*   **Twitter**: [@Proj_Chimera](https://x.com/Proj_Chimera  )
*   **Email**: projectchimera.dev@gmail.com

---

*Project Chimera is actively developed and maintained by the open-source community. Your contributions and feedback are invaluable to our mission of creating more robust, transparent, and reliable AI reasoning systems.*