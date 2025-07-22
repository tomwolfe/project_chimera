# Project Chimera

A lightweight Python application that runs a Socratic self-debate loop using a single LLM adopting different reasoning personas.

## Setup

1.  Clone this repository:
    ```bash
    git clone https://github.com/your-username/project-chimera.git
    cd project_chimera
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage (CLI)

Set your Gemini API Key as an environment variable:
```bash
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```
Alternatively, pass it directly: `--api-key YOUR_GEMINI_API_KEY`

Run the Socratic Arbitration Loop:
```bash
python main.py "Your prompt goes here, e.g., Design a sustainable city for 1 million people on Mars."
```

To see intermediate reasoning steps:
```bash
python main.py "Your prompt goes here." --verbose
```

## Usage (Web App - Streamlit)

1.  Set your Gemini API Key as an environment variable (recommended for local development):
    ```bash
    export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    ```
2.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
    Then open your web browser to the address provided by Streamlit (usually `http://localhost:8501`).