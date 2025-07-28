# app.py
import streamlit as st
import os
from core import run_isal_process
from core import TokenBudgetExceededError # Import new exception
# from core import Persona # Not directly used in app.py, but imported in core
import sys # Not used directly in app.py, but part of capture_rich_output
from rich.text import Text
from rich.syntax import Syntax
import io
import contextlib
from llm_provider import GeminiAPIError, LLMUnexpectedError # Import specific exceptions
from llm_provider import GeminiProvider # Import GeminiProvider to access cost calculation
import re # Added for stripping ANSI codes
import datetime # Added for timestamp in report
from typing import Dict, Any, Optional # Add this line
from collections import defaultdict
import yaml # Added for persona loading in app.py init
from rich.console import Console # <<< THIS IMPORT IS NECESSARY AND MUST BE PRESENT

# Redirect rich console output to a string buffer for Streamlit display
@contextlib.contextmanager
def capture_rich_output_and_get_console():
    """
    Captures rich console output to a StringIO buffer and returns the buffer
    along with the Console instance used for capturing.
    This avoids patching global state and is more robust.
    """
    # io.StringIO does NOT take an 'encoding' argument. It operates on Unicode strings directly.
    buffer = io.StringIO() 
    # Create a new Console instance that writes to the buffer.
    # Pass encoding="utf-8" to Console to ensure rich handles Unicode correctly when writing to the buffer.
    console_instance = Console(file=buffer, force_terminal=True, soft_wrap=True, encoding="utf-8")
    
    yield buffer, console_instance
    # No explicit restoration needed as this console instance is local to the context manager
    # and doesn't patch global rich Console instances or sys.stdout.

# Function to strip ANSI escape codes from text
ansi_escape_re = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
def strip_ansi_codes(text):
    return ansi_escape_re.sub('', text)

# --- Helper function for Markdown Report Generation ---
def generate_markdown_report(user_prompt: str, final_answer: str, intermediate_steps: Dict[str, Any], process_log_output: str, config_params: Dict[str, Any]) -> str:
    """
    Generates a comprehensive Markdown report of the Socratic Debate process.
    """
    report_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md_content = f"# Project Chimera Socratic Debate Report\n\n"
    md_content += f"**Date:** {report_date}\n"
    md_content += f"**Original Prompt:** {user_prompt}\n\n"

    md_content += "---\n\n"
    md_content += "## Configuration\n\n"
    md_content += f"*   **Model:** {config_params.get('model_name', 'N/A')}\n"
    md_content += f"*   **Max Total Tokens Budget:** {config_params.get('max_tokens_budget', 'N/A')}\n"
    md_content += f"*   **Intermediate Steps Shown in UI:** {'Yes' if config_params.get('show_intermediate_steps', False) else 'No'}\n"
    md_content += f"*   **Reasoning Framework:** {config_params.get('domain', 'N/A')}\n\n"


    md_content += "---\n\n"
    md_content += "## Process Log\n\n"
    md_content += "```text\n"
    md_content += strip_ansi_codes(process_log_output) # Strip ANSI codes for cleaner text file
    md_content += "\n```\n\n"

    if config_params.get('show_intermediate_steps', True): # Always include in full report if available
        md_content += "---\n\n"
        md_content += "## Intermediate Reasoning Steps\n\n"

        # Create a list of step names to process, excluding token counts and the total
        step_keys_to_process = [k for k in intermediate_steps.keys()
                                if not k.endswith("_Tokens_Used") and k != "Total_Tokens_Used" and k != "Total_Estimated_Cost_USD"]

        for step_key in step_keys_to_process: # step_key here is the content key
            # Determine the display name for the section
            display_name = step_key.replace('_Output', '').replace('_Critique', '').replace('_Feedback', '').replace('_', ' ').title()

            content = intermediate_steps.get(step_key, "N/A")

            # Find the corresponding token count key based on core.py's naming
            token_base_name = step_key
            if step_key.endswith("_Output"):
                token_base_name = step_key.replace("_Output", "")
            # For Skeptical_Critique, Constructive_Feedback, Devils_Advocate_Critique,
            # the step_key itself is the base for the token key.

            token_count_key = f"{token_base_name}_Tokens_Used"
            tokens_used = intermediate_steps.get(token_count_key, "N/A")

            md_content += f"### {display_name}\n\n"
            md_content += f"```markdown\n{content}\n```\n"
            md_content += f"**Tokens Used for this step:** {tokens_used}\n\n"

    md_content += "---\n\n"
    md_content += "## Final Synthesized Answer\n\n"
    md_content += f"{final_answer}\n\n" # Final answer is already markdown

    md_content += "---\n\n"
    md_content += "## Summary\n\n"
    md_content += f"**Total Tokens Consumed:** {intermediate_steps.get('Total_Tokens_Used', 0):,}\n"
    md_content += f"**Total Estimated Cost:** ${intermediate_steps.get('Total_Estimated_Cost_USD', 0.0):.4f}\n" # Add cost to report

    return md_content


st.set_page_config(layout="wide", page_title="Project Chimera Web App")

st.title("Project Chimera: Socratic Self-Debate")
st.markdown("Run an Iterative Socratic Arbitration Loop (ISAL) using a single LLM with multiple personas.")
st.markdown("This project's core software is open-source and available on [GitHub](https://github.com/tomwolfe/project_chimera).")

# Pre-defined Prompt Templates/Examples
EXAMPLE_PROMPTS = {
    "Design a Mars City": "Design a sustainable city for 1 million people on Mars, considering resource scarcity and human psychology.",
    "Ethical AI Framework": "Develop an ethical framework for an AI system designed to assist in judicial sentencing, addressing bias, transparency, and accountability.",
    "Future of Education": "Describe the future of education in 2050, incorporating AI, virtual reality, and personalized learning paths.",
    "Climate Change Solution": "Propose an innovative, scalable solution to mitigate the effects of climate change, focusing on a specific sector (e.g., energy, agriculture, transportation).",
    "Space Tourism Business Plan": "Outline a business plan for a luxury space tourism company, detailing target audience, unique selling propositions, and operational challenges.",
    # --- New Example Prompts ---
    "Future of Work": "Describe the future of work in a world increasingly dominated by automation and AI, considering new job roles, economic models, and the role of human creativity.",
    "Interstellar Travel Challenges": "Identify and propose solutions for the primary scientific and engineering challenges of achieving interstellar travel within the next 200 years.",
    "Ethical Considerations of Gene Editing": "Discuss the ethical implications of widespread human gene editing, considering accessibility, unintended consequences, and societal impact.",
    "Smart City Infrastructure": "Design a smart city infrastructure that leverages IoT, AI, and renewable energy to improve urban living, focusing on sustainability and citizen well-being.",
    "Personalized Learning Systems": "Develop a concept for a highly personalized learning system that adapts to individual student needs, learning styles, and career aspirations, incorporating AI and adaptive content.",
    "Global Pandemic Preparedness": "Outline a comprehensive global strategy for preventing and responding to future pandemics, including early warning systems, vaccine development, and equitable distribution.",
    "Sustainable Food Systems": "Propose innovative solutions for creating sustainable and resilient food systems that can feed a growing global population while minimizing environmental impact.",
    "AI in Healthcare Diagnostics": "Explore the potential and challenges of integrating AI into medical diagnostics, focusing on accuracy, data privacy, and the role of human practitioners.",
    "Revitalizing Rural Economies": "Develop a plan to revitalize rural economies in developed countries, considering remote work, local entrepreneurship, and sustainable tourism.",
    "Quantum Computing Impact": "Analyze the potential societal and economic impacts of widespread quantum computing by 2040, including both opportunities and risks.",
    "Personalized Medicine Ethics": "Discuss the ethical implications of highly personalized medicine, considering data privacy, equitable access, and genetic manipulation.",
    "Sustainable Urban Farming": "Detail a comprehensive plan for implementing sustainable urban farming on a large scale in a major metropolitan area, addressing infrastructure, community engagement, and economic viability.",
}

# --- Reset Function ---
def reset_app_state():
    """Resets all input fields and clears outputs."""
    st.session_state.api_key_input = ""
    st.session_state.user_prompt_input = EXAMPLE_PROMPTS["Design a Mars City"] # Reset to default example
    st.session_state.max_tokens_budget_input = 1000000
    st.session_state.show_intermediate_steps_checkbox = True
    st.session_state.selected_model_selectbox = "gemini-2.5-flash-lite"
    # When resetting, set example_selector to the key of the default prompt
    st.session_state.example_selector = "Design a Mars City"
    # Reset to default persona set
    try:
        import core
        _, persona_sets, default_set = core.load_personas()
        st.session_state.selected_persona_set = default_set
    except Exception: # Catch broader exceptions during file load
        st.session_state.selected_persona_set = "General"
    # Clear custom persona sets
    st.session_state.custom_persona_sets = {}

    # Clear all output-related session states
    st.session_state.debate_ran = False
    st.session_state.final_answer_output = ""
    st.session_state.intermediate_steps_output = {}
    st.session_state.process_log_output_text = ""
    st.session_state.last_config_params = {} # Clear last config params
    # Reset editable personas to default from file
    import core # Import here to avoid circular dependency at top if not already
    st.session_state.editable_personas = {p.name: p.model_dump() for p in core.load_personas()[0].values()} # Load initial personas as dicts for easy editing

    # Initialize domain-related states
    try:
        all_personas_from_file, persona_sets, default_set = core.load_personas() # Load all personas and sets
        st.session_state.available_domains = list(persona_sets.keys())
        st.session_state.domain = default_set
    except Exception:
        st.session_state.available_domains = ["General"]
        st.session_state.domain = "General"
    st.session_state.domain_recommendation = ""
    st.session_state.custom_persona_sets = {}
    st.session_state.selected_persona_set = st.session_state.domain # Ensure selected_persona_set is consistent


# Initialize all session state variables at the top
if "api_key_input" not in st.session_state:
    st.session_state.api_key_input = os.getenv("GEMINI_API_KEY", "")
if "user_prompt_input" not in st.session_state:
    st.session_state.user_prompt_input = EXAMPLE_PROMPTS["Design a Mars City"]
if "max_tokens_budget_input" not in st.session_state:
    st.session_state.max_tokens_budget_input = 1000000
if "show_intermediate_steps_checkbox" not in st.session_state:
    st.session_state.show_intermediate_steps_checkbox = True
if "selected_model_selectbox" not in st.session_state:
    st.session_state.selected_model_selectbox = "gemini-2.5-flash-lite"
# Initialize example_selector to the key of the default prompt
if "example_selector" not in st.session_state:
    st.session_state.example_selector = "Design a Mars City"

# New session state variables for outputs and control
if "debate_ran" not in st.session_state:
    st.session_state.debate_ran = False
if "available_domains" not in st.session_state:
    import core # Import core here for initial load
    try:
        _, persona_sets, default_set = core.load_personas()
        st.session_state.available_domains = list(persona_sets.keys())
        st.session_state.selected_persona_set = default_set
    except Exception:
        st.session_state.available_domains = ["General"]
        st.session_state.selected_persona_set = "General"
if "final_answer_output" not in st.session_state:
    st.session_state.final_answer_output = ""
if "intermediate_steps_output" not in st.session_state:
    st.session_state.intermediate_steps_output = {}
if "process_log_output_text" not in st.session_state:
    st.session_state.process_log_output_text = ""
if "last_config_params" not in st.session_state: # Store config for report generation
    st.session_state.last_config_params = {}
# New session state for editable personas
if "editable_personas" not in st.session_state: # Initialize only if not already present, using core.load_personas()[0] for all personas
    import core # Import here to avoid circular dependency at top if not already
    st.session_state.editable_personas = {p.name: p.model_dump() for p in core.load_personas()[0].values()} # Load initial personas as dicts for easy editing
if "custom_persona_sets" not in st.session_state:
    st.session_state.custom_persona_sets = {}


# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Configuration")

    # API Key Input
    st.text_input(
        "Enter your Gemini API Key",
        type="password",
        help="Your API key will not be stored. For deployed apps, consider using Streamlit Secrets (`st.secrets`).",
        key="api_key_input" # Value is implicitly taken from st.session_state.api_key_input
    )
    st.markdown("Need a Gemini API key? Get one from [Google AI Studio](https://aistudio.google.com/apikey).")

    st.markdown("---")

    # Model Selection
    st.selectbox(
        "Select LLM Model",
        ["gemini-2.5-flash-lite", "gemini-2.5-pro", "gemini-2.5-flash"],
        index=["gemini-2.5-flash-lite", "gemini-2.5-pro", "gemini-2.5-flash"].index(st.session_state.selected_model_selectbox), # Set index based on current model
        help="Choose the Gemini model based on your needs for speed, cost, or capability.",
        key="selected_model_selectbox" # Value is implicitly taken from st.session_state.selected_model_selectbox
    )

    st.markdown("---")

    # Max Tokens Budget
    st.number_input(
        "Max Total Tokens Budget:",
        min_value=1000,
        max_value=1000000, # Set a reasonable upper limit
        step=1000,
        help="Controls the maximum number of tokens used across all LLM calls to manage cost.",
        key="max_tokens_budget_input" # Value is implicitly taken from st.session_state.max_tokens_budget_input
    )
    # Placeholder for max tokens budget feedback
    max_tokens_feedback_placeholder = st.empty()
    if st.session_state.max_tokens_budget_input < 5000:
        max_tokens_feedback_placeholder.info("A budget below 5,000 tokens may result in an incomplete debate due to token limits.")
    else:
        max_tokens_feedback_placeholder.empty() # Clear warning if key is present

    st.markdown("---")

    # Show Intermediate Steps Checkbox
    st.checkbox(
        "Show Intermediate Reasoning Steps",
        key="show_intermediate_steps_checkbox" # Value is implicitly taken from st.session_state.show_intermediate_steps_checkbox
    )

# --- Main Content Area ---
st.header("Project Setup & Input")

# API Key Feedback (in main area for immediate visibility)
api_key_feedback_placeholder = st.empty()
if not st.session_state.api_key_input.strip():
    api_key_feedback_placeholder.warning("Please enter your Gemini API Key to enable the 'Run Socratic Debate' button.")
else:
    api_key_feedback_placeholder.empty() # Clear warning if key is present

# Callback function to update the prompt text area when an example is selected
def update_prompt_from_example():
    selected_example_name = st.session_state.example_selector
    if selected_example_name:
        st.session_state.user_prompt_input = EXAMPLE_PROMPTS[selected_example_name]

# Add Example Prompt Selector
st.selectbox(
    "Choose an example prompt:",
    options=[""] + list(EXAMPLE_PROMPTS.keys()), # Add an empty option for "no selection"
    help="Select a pre-defined prompt to quickly get started or see examples.",
    key="example_selector", # Key for the selectbox
    on_change=update_prompt_from_example # Callback to run when selection changes
)

# Prompt Input
user_prompt = st.text_area(
    "Enter your prompt here:",
    height=150,
    key="user_prompt_input" # Value is implicitly taken from st.session_state.user_prompt_input
)

st.markdown("---") # Divider

# --- Reasoning Framework Selection ---
st.subheader("Reasoning Framework Selection")

# Domain keyword mapping for prompt analysis
DOMAIN_KEYWORDS = {
    "Science": ["scientific", "research", "experiment", "data analysis", "hypothesis", "theory", "biology", "physics", "chemistry", "astronomy", "engineering", "algorithm", "computation", "quantum", "genetics", "ecology", "neuroscience", "medical", "diagnostic", "clinical", "biotech"],
    "Business": ["business", "market", "strategy", "finance", "investment", "startup", "profit", "revenue", "marketing", "sales", "operations", "management", "economy", "entrepreneurship", "product", "customer", "competitor", "plan", "model", "growth", "tourism"],
    "Creative": ["creative", "art", "story", "design", "narrative", "fiction", "poetry", "music", "film", "painting", "sculpture", "writing", "imagination", "concept", "aesthetic", "expression", "character", "plot", "world-building"]
}

def analyze_prompt_for_domain(prompt: str) -> Optional[str]:
    """Analyzes the prompt to suggest a domain based on keywords."""
    prompt_lower = prompt.lower()
    scores = defaultdict(int)

    for domain, keywords in DOMAIN_KEYWORDS.items():
        for keyword in keywords:
            if keyword in prompt_lower:
                scores[domain] += 1

    if not scores:
        return None

    # Get the domain with the highest score
    suggested_domain = max(scores.items(), key=lambda x: x[1])[0]
    return suggested_domain if scores[suggested_domain] > 0 else None

def get_domain_recommendation(prompt: str, api_key: str) -> str:
    """Gets a domain recommendation using both keyword analysis and LLM analysis."""
    # First try keyword analysis (faster, no API call)
    keyword_domain = analyze_prompt_for_domain(prompt)

    # If keyword analysis is inconclusive, use LLM analysis
    if not keyword_domain or keyword_domain not in st.session_state.available_domains:
        try:
            from llm_provider import recommend_domain
            llm_domain = recommend_domain(prompt, api_key)
            if llm_domain in st.session_state.available_domains:
                return llm_domain
        except Exception: # Catch any error from LLM recommendation
            pass

    # Return keyword domain if available and valid
    if keyword_domain and keyword_domain in st.session_state.available_domains:
        return keyword_domain

    return "General"

# Display domain recommendation
if user_prompt.strip() and st.session_state.api_key_input.strip():
    suggested_domain = get_domain_recommendation(user_prompt, st.session_state.api_key_input)
    if suggested_domain and suggested_domain != st.session_state.selected_persona_set:
        st.info(f"💡 Based on your prompt, we recommend using the **'{suggested_domain}'** reasoning framework.")
        if st.button(f"Apply '{suggested_domain}' Framework", type="primary", use_container_width=True):
            st.session_state.selected_persona_set = suggested_domain
            st.rerun()

# Persona Set Selection
st.selectbox(
    "Select Reasoning Framework",
    options=st.session_state.available_domains + ["Custom"],
    index=st.session_state.available_domains.index(st.session_state.selected_persona_set) if st.session_state.selected_persona_set in st.session_state.available_domains else len(st.session_state.available_domains),
    help="Choose a domain-specific reasoning framework that best matches your problem type.",
    key="selected_persona_set"
)

# Community Persona Sets
if st.session_state.custom_persona_sets:
    st.markdown("### Community Frameworks")
    community_col1, community_col2 = st.columns([3, 1])
    with community_col1:
        community_sets = list(st.session_state.custom_persona_sets.keys())
        selected_community_set = st.selectbox(
            "Community Frameworks",
            options=community_sets,
            help="Choose from community-contributed reasoning frameworks"
        )
    with community_col2:
        if st.button("Apply", type="primary", use_container_width=True):
            import core
            from core import Persona
            # Load the full set of personas from file to ensure all base personas are available
            all_base_personas, _, _ = core.load_personas()

            # Merge custom personas with base personas, prioritizing custom ones
            merged_personas = all_base_personas.copy()
            for name, data in st.session_state.custom_persona_sets[selected_community_set].items():
                merged_personas[name] = Persona(**data)

            st.session_state.selected_persona_set = selected_community_set
            st.session_state.editable_personas = {p.name: p.model_dump() for p in merged_personas.values()}
            st.success(f"Applied framework: {selected_community_set}")
            st.rerun()

# Add new community framework
with st.expander("Contribute Your Framework"):
    st.markdown("Save your current persona configuration as a reusable framework:")
    framework_name = st.text_input("Framework Name", help="Name your framework for community sharing")
    if st.button("Save as Community Framework", type="primary"):
        if not framework_name:
            st.error("Please provide a name for your framework")
        else:
            import core
            from core import Persona
            custom_personas = {name: Persona(**data) for name, data in st.session_state.editable_personas.items()}
            st.session_state.custom_persona_sets[framework_name] = custom_personas
            st.success(f"Framework '{framework_name}' saved to community library!")
            st.info("This framework will only persist for your current session. For permanent contributions, please submit a GitHub pull request.")

st.markdown("---")

# --- Persona Configuration UI ---
st.subheader("Persona Configuration")
with st.expander("View and Edit Personas"):
    persona_config_is_valid = True # Reset validation flag for this run
    st.markdown("Adjust the system prompt, temperature, and max tokens for each persona. Changes are temporary for this session.")
    # If a custom set is selected, load those specific personas for editing
    if st.session_state.selected_persona_set == "Custom":
        st.warning("You are in 'Custom' framework mode. Personas displayed below are the ones currently loaded. You can edit them, or save them as a new 'Community Framework'.")
        personas_to_display = st.session_state.editable_personas
    else:
        # Load the specific persona set from file for display/editing
        import core
        all_personas_from_file, persona_sets_from_file, _ = core.load_personas()

        # Get the names of personas in the selected set
        persona_names_in_set = persona_sets_from_file.get(st.session_state.selected_persona_set, [])

        # Filter all_personas_from_file to only include those in the selected set
        personas_to_display = {name: all_personas_from_file[name].model_dump() for name in persona_names_in_set if name in all_personas_from_file}

        # Update editable_personas session state to reflect the selected set
        # This ensures that when the user clicks "Run", the correct set is used.
        st.session_state.editable_personas = personas_to_display


    for persona_name, persona_data in personas_to_display.items():
        # Use a nested expander for each persona's details
        with st.expander(f"**{persona_name}** - {persona_data.get('description', 'No description provided.')}"):
            # System Prompt
            # Ensure the value comes from the session state if it's already there (for edits)
            # Use a unique key for each text_area based on persona_name
            current_system_prompt = st.session_state.editable_personas.get(persona_name, {}).get("system_prompt", persona_data.get("system_prompt", ""))
            st.session_state.editable_personas[persona_name]["system_prompt"] = st.text_area(
                f"System Prompt for {persona_name}:",
                value=current_system_prompt,
                height=150,
                key=f"system_prompt_{persona_name}"
            )
            # Update session state immediately with the value from the widget
            st.session_state.editable_personas[persona_name]["system_prompt"] = st.session_state[f"system_prompt_{persona_name}"]

            if not st.session_state.editable_personas[persona_name]["system_prompt"].strip():
                st.error(f"System prompt for '{persona_name}' cannot be empty.")
                persona_config_is_valid = False

            # Temperature and Max Tokens
            col_temp, col_max_tokens = st.columns(2)
            with col_temp:
                # Use a unique key for each slider
                current_temperature = st.session_state.editable_personas.get(persona_name, {}).get("temperature", persona_data.get("temperature", 0.5))
                st.session_state.editable_personas[persona_name]["temperature"] = st.slider(
                    f"Temperature for {persona_name}:",
                    min_value=0.0,
                    max_value=1.0,
                    value=current_temperature,
                    step=0.05,
                    key=f"temperature_{persona_name}"
                )
                # Update session state immediately
                st.session_state.editable_personas[persona_name]["temperature"] = st.session_state[f"temperature_{persona_name}"]
            with col_max_tokens:
                # Use a unique key for each number_input
                current_max_tokens = st.session_state.editable_personas.get(persona_name, {}).get("max_tokens", persona_data.get("max_tokens", 1024))
                st.session_state.editable_personas[persona_name]["max_tokens"] = st.number_input(
                    f"Max Tokens for {persona_name}:",
                    min_value=128, # Reasonable minimum for persona output
                    max_value=4096, # Reasonable maximum for persona output
                    value=current_max_tokens,
                    step=128,
                    key=f"max_tokens_{persona_name}"
                )
                # Update session state immediately
                st.session_state.editable_personas[persona_name]["max_tokens"] = st.session_state[f"max_tokens_{persona_name}"]

            # If any persona has an invalid config, mark the overall config as invalid
            if not st.session_state.editable_personas[persona_name]["system_prompt"].strip():
                persona_config_is_valid = False


# Run and Reset Buttons
run_col, reset_col = st.columns([0.7, 0.3]) # Adjust column width for buttons
with run_col:
    run_button_clicked = st.button("Run Socratic Debate", type="primary")
with reset_col:
    st.button("Reset All Inputs & Outputs", on_click=reset_app_state)


if run_button_clicked:
    # Clear any proactive feedback messages when run button is clicked
    api_key_feedback_placeholder.empty()
    max_tokens_feedback_placeholder.empty()

    if not st.session_state.api_key_input.strip():
        st.error("Please enter your Gemini API Key to proceed.")
    elif not user_prompt.strip():
        st.error("Please enter a prompt.")
    elif not persona_config_is_valid:
        st.error("Please correct the errors in the Persona Configuration before running the debate.")
    else:
        # Reset output states at the start of a new run
        st.session_state.debate_ran = False # Set to False initially, then True on success/error
        st.session_state.final_answer_output = ""
        st.session_state.intermediate_steps_output = {}
        st.session_state.process_log_output_text = ""
        st.session_state.last_config_params = {}

        # Convert dicts in session_state.editable_personas back to Persona objects for core.py
        import core # Import Persona class from core

        # Use st.status for real-time feedback during the process
        with st.status("Initializing Socratic Debate...", expanded=True) as status:
            # Placeholders for real-time metrics
            st.markdown("---")
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            total_tokens_placeholder = metric_col1.empty()
            total_cost_placeholder = metric_col2.empty()
            next_step_warning_placeholder = metric_col3.empty()
            st.markdown("---")

            # Define the callback function to update Streamlit UI elements
            def streamlit_status_callback(message: str, state: str = "running", expanded: bool = True,
                                          current_total_tokens: int = 0, current_total_cost: float = 0.0,
                                          estimated_next_step_tokens: int = 0, estimated_next_step_cost: float = 0.0):
                status.update(label=message, state=state, expanded=expanded)
                total_tokens_placeholder.metric("Total Tokens Used", f"{current_total_tokens:,}")
                total_cost_placeholder.metric("Estimated Cost (USD)", f"${current_total_cost:.4f}")

                # Proactive warning for next step
                if estimated_next_step_tokens > 0:
                    budget_remaining = st.session_state.max_tokens_budget_input - current_total_tokens
                    if estimated_next_step_tokens > budget_remaining:
                        next_step_warning_placeholder.warning(
                            f"⚠️ Next step ({estimated_next_step_tokens:,} tokens) "
                            f"will exceed budget ({budget_remaining:,} remaining). "
                            f"Estimated cost: ${estimated_next_step_cost:.4f}"
                        )
                    else:
                        next_step_warning_placeholder.info(
                            f"Next step estimated: {estimated_next_step_tokens:,} tokens "
                            f"(${(estimated_next_step_cost):.4f}). "
                            f"Budget remaining: {budget_remaining:,} tokens."
                        )
                else:
                    next_step_warning_placeholder.empty() # Clear warning if no next step estimate

            # Use the context manager to capture rich output for the log display
            with capture_rich_output_and_get_console() as (rich_output_buffer, rich_console_instance):
                # Initialize debate_instance before the try block
                debate_instance = None
                try:
                    # First, get the debate instance
                    from core import SocraticDebate # Import SocraticDebate class

                    # Determine which personas to use: custom or a predefined set
                    if st.session_state.selected_persona_set == "Custom":
                        personas_to_use = {name: core.Persona(**data) for name, data in st.session_state.editable_personas.items()}
                        domain_for_report = "Custom"
                        all_personas_for_core = {name: core.Persona(**data) for name, data in st.session_state.editable_personas.items()} # If custom, these are the 'all'
                        persona_sets_for_core = {"Custom": list(personas_to_use.keys())} # Define a custom set for core
                    else:
                        # Load all personas and sets from file
                        all_personas_from_file, persona_sets_from_file, default_set = core.load_personas()
                        # Get the names of personas in the selected set
                        persona_names_in_set = persona_sets_from_file.get(st.session_state.selected_persona_set, [])
                        # Create Persona objects for the selected set (these are the ones to be used in the debate flow)
                        personas_to_use = {name: all_personas_from_file[name] for name in persona_names_in_set if name in all_personas_from_file}
                        domain_for_report = st.session_state.selected_persona_set
                        all_personas_for_core = all_personas_from_file # Pass the full dictionary
                        persona_sets_for_core = persona_sets_from_file # Pass the full dictionary

                    debate_instance: SocraticDebate = run_isal_process(
                        user_prompt, st.session_state.api_key_input, max_total_tokens_budget=st.session_state.max_tokens_budget_input,
                        streamlit_status_callback=streamlit_status_callback, # Pass the callback
                        model_name=st.session_state.selected_model_selectbox, # Pass the selected model name
                        domain=domain_for_report, # Pass the selected domain
                        personas_override=personas_to_use, # Pass the edited/selected personas
                        all_personas=all_personas_for_core, # Pass all personas to core
                        persona_sets=persona_sets_for_core, # Pass all persona sets to core
                        rich_console=rich_console_instance # Pass the rich console instance
                    )
                    # Then, run the debate process
                    final_answer, intermediate_steps = debate_instance.run_debate()

                    st.session_state.process_log_output_text = rich_output_buffer.getvalue()
                    st.session_state.final_answer_output = final_answer
                    st.session_state.intermediate_steps_output = intermediate_steps
                    st.session_state.last_config_params = {
                        "max_tokens_budget": st.session_state.max_tokens_budget_input,
                        "model_name": st.session_state.selected_model_selectbox,
                        "show_intermediate_steps": st.session_state.show_intermediate_steps_checkbox, # Corrected reference
                        "domain": domain_for_report # Store domain for report
                    }
                    st.session_state.debate_ran = True

                    # Update status to complete after successful execution
                    status.update(label="Socratic Debate Complete!", state="complete", expanded=False)
                    # Ensure final metrics are displayed
                    final_total_tokens = intermediate_steps.get('Total_Tokens_Used', 0)
                    final_total_cost = intermediate_steps.get('Total_Estimated_Cost_USD', 0.0)

                    total_tokens_placeholder.metric("Total Tokens Used", f"{final_total_tokens:,}")
                    total_cost_placeholder.metric("Estimated Cost (USD)", f"${final_total_cost:.4f}")
                    next_step_warning_placeholder.empty() # Clear any pending warnings

                except TokenBudgetExceededError as e: # Catch the new specific error
                    st.session_state.process_log_output_text = rich_output_buffer.getvalue()
                    # Access intermediate_steps from the debate_instance
                    st.session_state.intermediate_steps_output = debate_instance.intermediate_steps if debate_instance else {}
                    st.session_state.last_config_params = { # Capture config even on error
                        "max_tokens_budget": st.session_state.max_tokens_budget_input,
                        "model_name": st.session_state.selected_model_selectbox,
                        "show_intermediate_steps": st.session_state.show_intermediate_steps_checkbox, # Corrected reference
                        "domain": st.session_state.selected_persona_set # Store domain for report
                    }
                    st.session_state.debate_ran = True # Still show error output
                    error_message = str(e)
                    user_advice = "The process was stopped because it would exceed the maximum token budget."
                    status.update(label=f"Socratic Debate Failed: {user_advice}", state="error", expanded=True)
                    st.error(f"**Error:** {user_advice}\n\n**Details:** {error_message}")
                    # Ensure final metrics are displayed even on error
                    total_tokens_placeholder.metric("Total Tokens Used", f"{debate_instance.intermediate_steps.get('Total_Tokens_Used', 0):,}" if debate_instance else "N/A")
                    total_cost_placeholder.metric("Estimated Cost (USD)", f"${debate_instance.intermediate_steps.get('Total_Estimated_Cost_USD', 0.0):.4f}" if debate_instance else "N/A")

                except GeminiAPIError as e:
                    st.session_state.process_log_output_text = rich_output_buffer.getvalue()
                    st.session_state.intermediate_steps_output = debate_instance.intermediate_steps if debate_instance else {}
                    st.session_state.last_config_params = {
                        "max_tokens_budget": st.session_state.max_tokens_budget_input,
                        "model_name": st.session_state.selected_model_selectbox,
                        "show_intermediate_steps": st.session_state.show_intermediate_steps_checkbox, # Corrected reference
                        "domain": st.session_state.selected_persona_set # Store domain for report
                    }
                    st.session_state.debate_ran = True
                    error_message = str(e) # Get the message from the exception object
                    user_advice = "An issue occurred with the Gemini API."
                    if e.code == 401 or "invalid API key" in error_message.lower():
                        user_advice = "Invalid API Key. Please check your Gemini API key and ensure it is correct."
                    elif e.code == 429 or "rate limit" in error_message.lower():
                        user_advice = "Rate Limit Exceeded. You've made too many requests. Please wait a moment and try again."
                    elif e.code == 403 or "quota" in error_message.lower():
                        user_advice = "Quota Exceeded or Access Denied. Please check your Gemini API quota or permissions."
                    elif "model" in error_message.lower() and "not found" in error_message.lower():
                        user_advice = f"Selected model '{st.session_state.selected_model_selectbox}' not found or not available. Please choose a different model."

                    status.update(label=f"Socratic Debate Failed: {user_advice}", state="error", expanded=True)
                    st.error(f"**Error:** {user_advice}\n\n**Details:** {error_message}")
                    # Ensure final metrics are displayed even on error
                    total_tokens_placeholder.metric("Total Tokens Used", f"{debate_instance.intermediate_steps.get('Total_Tokens_Used', 0):,}" if debate_instance else "N/A")
                    total_cost_placeholder.metric("Estimated Cost (USD)", f"${debate_instance.intermediate_steps.get('Total_Estimated_Cost_USD', 0.0):.4f}" if debate_instance else "N/A")

                except LLMUnexpectedError as e:
                    st.session_state.process_log_output_text = rich_output_buffer.getvalue()
                    st.session_state.intermediate_steps_output = debate_instance.intermediate_steps if debate_instance else {}
                    st.session_state.last_config_params = {
                        "max_tokens_budget": st.session_state.max_tokens_budget_input,
                        "model_name": st.session_state.selected_model_selectbox,
                        "show_intermediate_steps": st.session_state.show_intermediate_steps_checkbox, # Corrected reference
                        "domain": st.session_state.selected_persona_set # Store domain for report
                    }
                    st.session_state.debate_ran = True
                    error_message = str(e)
                    user_advice = "An unexpected issue occurred with the LLM provider (e.g., network problem, malformed response). Please try again later."

                    status.update(label=f"Socratic Debate Failed: {user_advice}", state="error", expanded=True)
                    st.error(f"**Error:** {user_advice}\n\n**Details:** {error_message}")
                    # Ensure final metrics are displayed even on error
                    total_tokens_placeholder.metric("Total Tokens Used", f"{debate_instance.intermediate_steps.get('Total_Tokens_Used', 0):,}" if debate_instance else "N/A")
                    total_cost_placeholder.metric("Estimated Cost (USD)", f"${debate_instance.intermediate_steps.get('Total_Estimated_Cost_USD', 0.0):.4f}" if debate_instance else "N/A")

                except Exception as e:
                    st.session_state.process_log_output_text = rich_output_buffer.getvalue()
                    # For a generic Exception, ensure debate_instance is available
                    if debate_instance is not None: # Check if debate_instance was successfully created
                        st.session_state.intermediate_steps_output = debate_instance.intermediate_steps
                        total_tokens_val = debate_instance.intermediate_steps.get("Total_Tokens_Used", 0)
                        total_cost_val = debate_instance.intermediate_steps.get("Total_Estimated_Cost_USD", 0.0)
                    else:
                        st.session_state.intermediate_steps_output = {} # Fallback if debate_instance itself failed to create
                        total_tokens_val = 0
                        total_cost_val = 0.0

                    st.session_state.last_config_params = {
                        "max_tokens_budget": st.session_state.max_tokens_budget_input,
                        "model_name": st.session_state.selected_model_selectbox,
                        "show_intermediate_steps": st.session_state.show_intermediate_steps_checkbox, # Corrected reference
                        "domain": st.session_state.selected_persona_set # Store domain for report
                    }
                    st.session_state.debate_ran = True
                    # Update status to error if an exception occurs
                    status.update(label=f"Socratic Debate Failed: {e}", state="error", expanded=True)
                    st.error(f"An unexpected error occurred during the process: {e}")
                    # Ensure final metrics are displayed even on error
                    total_tokens_placeholder.metric("Total Tokens Used", f"{total_tokens_val:,}")
                    total_cost_placeholder.metric("Estimated Cost (USD)", f"${total_cost_val:.4f}")

# This block will only render if a debate has been run (successfully or with error)
if st.session_state.debate_ran:
    st.subheader("Process Log")
    st.code(strip_ansi_codes(st.session_state.process_log_output_text), language="text")

    if st.session_state.show_intermediate_steps_checkbox: # Use session state for checkbox value
        st.subheader("Intermediate Reasoning Steps")
        # Filter out Total_Tokens_Used and Total_Estimated_Cost_USD for this display, it's shown separately
        display_steps = {k: v for k, v in st.session_state.intermediate_steps_output.items()
                         if k not in ["Total_Tokens_Used", "Total_Estimated_Cost_USD"]}

        # Group step outputs with their token counts for display
        processed_keys = set()
        for content_key, content in display_steps.items(): # content_key is like 'Visionary_Generator_Output', 'Skeptical_Critique'
            if content_key.endswith("_Tokens_Used") or content_key in processed_keys:
                continue

            # Determine the display name for the expander
            display_name = content_key.replace('_Output', '').replace('_Critique', '').replace('_Feedback', '').replace('_', ' ').title()

            content = intermediate_steps.get(content_key, "N/A")

            # Find the corresponding token count key based on core.py's naming
            token_base_name = content_key
            if content_key.endswith("_Output"):
                token_base_name = content_key.replace("_Output", "")
            # For Skeptical_Critique, Constructive_Feedback, Devils_Advocate_Critique,
            # the content_key itself is the base for the token key.

            token_count_key = f"{token_base_name}_Tokens_Used"
            tokens_used = intermediate_steps.get(token_count_key, "N/A")

            with st.expander(f"### {display_name}"):
                st.code(content, language="markdown")
                st.write(f"**Tokens used for this step:** {tokens_used}")

            processed_keys.add(content_key)
            processed_keys.add(token_count_key)

    st.subheader("Final Synthesized Answer")
    st.markdown(st.session_state.final_answer_output) # Use session state for final answer

    # --- Export Functionality ---
    st.markdown("---")
    st.subheader("Export Results")

    # Use the prompt from the text area, as it's the current state of the prompt input
    final_answer_md = f"# Final Synthesized Answer\n\n{st.session_state.final_answer_output}"
    st.download_button(
        label="Download Final Answer (Markdown)",
        data=final_answer_md,
        file_name="final_answer.md",
        mime="text/markdown"
    )

    full_report_md = generate_markdown_report(
        user_prompt=user_prompt, # This is the current user_prompt from the text_area
        final_answer=st.session_state.final_answer_output,
        intermediate_steps=st.session_state.intermediate_steps_output,
        process_log_output=st.session_state.process_log_output_text,
        config_params=st.session_state.last_config_params # Use stored config
    )
    st.download_button(
        label="Download Full Report (Markdown)",
        data=full_report_md,
        file_name="socratic_debate_report.md",
        mime="text/markdown"
    )
    st.info("To generate a PDF, download the Markdown report and use your browser's 'Print to PDF' option (usually accessible via Ctrl+P or Cmd+P).")