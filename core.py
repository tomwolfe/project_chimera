# core.py
# core.py
import yaml
import time
import hashlib
import sys
import re
import ast
import pycodestyle
import difflib
import subprocess
import tempfile
import os
import json
from rich.console import Console
from pydantic import BaseModel, Field, ValidationError, model_validator, validator
import streamlit as st # For st.cache_resource/data decorators
from typing import List, Dict, Tuple, Any, Callable, Optional
# Import LLMProvider and LLMOutputParser
from llm_provider import GeminiProvider, LLMProviderError, GeminiAPIError, LLMUnexpectedError
from src.utils.output_parser import LLMOutputParser
from src.utils.code_validator import validate_code_output_batch # Import the batch validation function
# Import Pydantic models for Persona configuration
from src.models import PersonaConfig, ReasoningFrameworkConfig
# --- Custom Exception for Token Budget ---
class TokenBudgetExceededError(LLMProviderError):
    """Raised when an LLM call would exceed the total token budget."""
    pass
@st.cache_resource
def load_personas(file_path: str = 'personas.yaml') -> Tuple[Dict[str, PersonaConfig], Dict[str, List[str]], List[str], str]:
    """Loads persona configurations from a YAML file. Cached using st.cache_resource."""
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        # Manually parse and validate to handle the structure not directly matching FullPersonaConfig
        # The personas.yaml has 'personas' as a list of dicts, and 'persona_sets' as a dict of lists.
        # We need to convert the list of persona dicts into a dict of PersonaConfig objects.
        all_personas_list = [PersonaConfig(**p_data) for p_data in data.get('personas', [])]
        all_personas_dict = {p.name: p for p in all_personas_list}
        persona_sets = data.get('persona_sets', {"General": []})
        persona_sequence = data.get('persona_sequence', [
            "Visionary_Generator",
            "Skeptical_Generator",
            "Constructive_Critic",
            "Impartial_Arbitrator",
            "Devils_Advocate"
        ])
        # Validate references in persona_sets and persona_sequence
        for set_name, persona_names_in_set in persona_sets.items():
            if not isinstance(persona_names_in_set, list):
                raise ValueError(f"Persona set '{set_name}' must be a list of persona names.")
            for p_name in persona_names_in_set:
                if p_name not in all_personas_dict:
                    raise ValueError(f"Persona '{p_name}' referenced in set '{set_name}' not found in 'personas' list.")
        for p_name in persona_sequence:
            if p_name not in all_personas_dict:
                raise ValueError(f"Persona '{p_name}' in persona_sequence not found in 'personas' list.")
        default_persona_set_name = "General" if "General" in persona_sets else next(iter(persona_sets.keys()))
        return all_personas_dict, persona_sets, persona_sequence, default_persona_set_name
    except (FileNotFoundError, ValidationError, yaml.YAMLError) as e:
        logging.error(f"Error loading personas from {file_path}: {e}")
        raise
class SocraticDebate:
    DEFAULT_MAX_RETRIES = 2
    MAX_BACKOFF_SECONDS = 30
    def __init__(self,
                 initial_prompt: str,
                 api_key: str,
                 max_total_tokens_budget: int,
                 model_name: str,
                 personas: Dict[str, PersonaConfig],
                 all_personas: Dict[str, PersonaConfig],
                 persona_sets: Dict[str, List[str]],
                 persona_sequence: List[str],
                 gemini_provider: Optional[GeminiProvider] = None,
                 domain: str = "General",
                 status_callback: Callable = None,
                 rich_console: Optional[Console] = None,
                 codebase_context: Optional[Dict[str, str]] = None,
                 context_token_budget_ratio: float = 0.25):
        self.initial_prompt = initial_prompt
        self.max_total_tokens_budget = max_total_tokens_budget
        self.model_name = model_name
        self.personas = personas
        self.domain = domain
        self.all_personas = all_personas
        self.persona_sets = persona_sets
        self.persona_sequence = persona_sequence
        self.status_callback = status_callback
        self.context_token_budget_ratio = context_token_budget_ratio
        if gemini_provider:
            self.gemini_provider = gemini_provider
        else:
            self.gemini_provider = GeminiProvider(api_key=api_key, model_name=model_name,
                                                  _status_callback=self._update_status) # Use internal callback
        # Initialize LLMOutputParser
        self.parser = LLMOutputParser(self.gemini_provider)
        self.cumulative_token_usage = 0
        self.cumulative_usd_cost = 0.0
        self.intermediate_steps: Dict[str, Any] = {}
        self.rich_console = rich_console if rich_console else Console()
        self.current_thought = initial_prompt
        self.final_answer = "Process did not complete." # Default value
        self.codebase_context = codebase_context
    def _update_status(self, message: str, **kwargs):
        """Helper to print to console and call Streamlit callback."""
        self.rich_console.print(message)
        if self.status_callback:
            self.status_callback(message=message, **kwargs)
    def _get_persona(self, name: str) -> PersonaConfig:
        """Retrieves a persona by name, checking both specific and all personas."""
        persona = self.personas.get(name) or self.all_personas.get(name)
        if not persona:
            raise ValueError(f"Persona '{name}' not found.")
        return persona
    @st.cache_data(ttl=3600)
    def _prioritize_python_code(content: str, max_tokens: int, gemini_provider: GeminiProvider) -> str:
        """
        Prioritizes imports, class/function definitions for Python code.
        Truncates the content to fit within max_tokens.
        """
        lines = content.splitlines()
        priority_lines = []
        other_lines = []
        try:
            tree = ast.parse(content)
            priority_line_numbers = set()
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef)):
                    start_lineno = node.lineno - 1
                    end_lineno = node.end_lineno if hasattr(node, 'end_lineno') else start_lineno + 1
                    for i in range(start_lineno, end_lineno):
                        if i < len(lines):
                            priority_line_numbers.add(i)
            for i, line in enumerate(lines):
                if i in priority_line_numbers:
                    priority_lines.append(line)
                else:
                    other_lines.append(line)
        except SyntaxError:
            # Log this warning, but don't use self._update_status as it's a static method
            logging.warning(f"[yellow]Warning: Syntax error in Python context file, falling back to simple truncation.[/yellow]")
            return SocraticDebate._truncate_text_by_tokens(content, max_tokens, gemini_provider)
        combined_content = "\n".join(priority_lines + other_lines)
        return SocraticDebate._truncate_text_by_tokens(combined_content, max_tokens, gemini_provider)
    def _truncate_text_by_tokens(self, text: str, max_tokens: int) -> str:
        """Truncates text to fit within max_tokens using the GeminiProvider's token counting."""
        if not text:
            return ""
        current_tokens = self.gemini_provider.count_tokens(text, "")
        if current_tokens <= max_tokens:
            return text
        chars_per_token_estimate = 4
        target_chars = max_tokens * chars_per_token_estimate
        truncated_text = text
        if len(truncated_text) > target_chars:
            truncated_text = truncated_text[:target_chars]
        while self.gemini_provider.count_tokens(truncated_text, "") > max_tokens and len(truncated_text) > 0:
            chars_to_remove = max(1, len(truncated_text) // 20)
            truncated_text = truncated_text[:-chars_to_remove]
            if len(truncated_text) == 0:
                break
        if self.gemini_provider.count_tokens(text, "") > max_tokens:
            return truncated_text.strip() + "\n... (truncated)"
        return truncated_text
    def prepare_context(self) -> str:
        """Prepares the codebase context, prioritizing Python code and truncating to fit budget."""
        if not self.codebase_context:
            return "No codebase context provided."
        context_budget = int(self.max_total_tokens_budget * self.context_token_budget_ratio)
        context_str_parts = []
        current_tokens = 0
        for path, content in self.codebase_context.items():
            header = f"--- file_path: {path} ---\n"
            header_tokens = self.gemini_provider.count_tokens(header, "")
            remaining_budget_for_file_content = context_budget - current_tokens - header_tokens
            if remaining_budget_for_file_content <= 0:
                self._update_status(f"Skipping file '{path}' due to context token budget exhaustion.")
                break
            file_content_to_add = ""
            if path.endswith('.py'):
                prioritized_content = SocraticDebate._prioritize_python_code(content, remaining_budget_for_file_content, self.gemini_provider)
                file_content_to_add = self._truncate_text_by_tokens(prioritized_content, remaining_budget_for_file_content)
            else:
                file_content_to_add = self._truncate_text_by_tokens(content, remaining_budget_for_file_content)
            if not file_content_to_add:
                continue
            full_file_block = header + file_content_to_add + "\n"
            file_block_tokens = self.gemini_provider.count_tokens(full_file_block, "")
            if current_tokens + file_block_tokens > context_budget:
                self._update_status(f"Warning: Could not fit full file '{path}' even after truncation. Skipping remaining files.")
                break
            context_str_parts.append(full_file_block)
            current_tokens += file_block_tokens
        final_context_string = "".join(context_str_parts)
        self._update_status(f"Prepared codebase context using {self.gemini_provider.count_tokens(final_context_string, '')} tokens.")
        return final_context_string
    def _analyze_codebase_context(self) -> Dict[str, Any]:
        """
        Analyzes the codebase context using a dedicated persona.
        CRE_001: Introduce a 'Context Analysis' step using a dedicated persona.
        """
        if not self.codebase_context:
            self._update_status("[yellow]No codebase context provided for analysis.[/yellow]")
            return self._get_default_analysis_summary()
        # Check if Context_Aware_Assistant persona is available
        if "Context_Aware_Assistant" not in self.personas and "Context_Aware_Assistant" not in self.all_personas:
            self._update_status("[yellow]Context_Aware_Assistant persona not found. Skipping context analysis.[/yellow]")
            return self._get_default_analysis_summary()
        self._update_status("Analyzing codebase context with Context_Aware_Assistant...")
        # Prepare the context string for the analysis persona
        context_string_for_analysis = self.prepare_context() # Use context preparation logic
        codebase_analysis_summary = self._analyze_codebase_context()
        analysis_str = json.dumps(codebase_analysis_summary, indent=2)
        def visionary_prompt_gen():
            if codebase_analysis_summary:
                return (f"USER PROMPT: {self.initial_prompt}\n\nCODEBASE CONTEXT ANALYSIS:\n{analysis_str}\n\n"
                        f"CODEBASE CONTEXT:\n{context_string_for_analysis}\n\nINSTRUCTIONS:\n"
                        f"1. **Analyze the provided codebase context and its analysis thoroughly.** Understand its structure, style, patterns, dependencies, and overall logic, incorporating insights from the analysis.\n"
                        f"2. **Propose an initial implementation strategy or code snippet.** Your proposal should be consistent with the existing codebase and address any identified concerns from the analysis.\n"
                        f"3. **Ensure your proposed code fits naturally into the existing architecture and follows its conventions.** Use the provided `GeminiProvider` and `SocraticDebate` classes as examples of how to integrate.")
            return (f"USER PROMPT: {self.initial_prompt}\n\n"
                    f"CODEBASE CONTEXT:\n{context_string_for_analysis}\n\n"
                    f"INSTRUCTIONS:\n"
                    f"1. **Analyze the provided codebase context thoroughly.** Understand its structure, style, patterns, dependencies, and overall logic.\n"
                    f"2. **Propose an initial implementation strategy or code snippet.** Your proposal should be consistent with the existing codebase.\n"
                    f"3. **Ensure your proposed code fits naturally into the existing architecture and follows its conventions.** Use the provided `GeminiProvider` and `SocraticDebate` classes as examples of how to integrate.")
        analysis_output_key = "Context_Analysis_Output"
        try:
            # Execute the persona step. We expect a string response which is JSON.
            raw_analysis_response = self._execute_persona_step(
                "Context_Aware_Assistant",
                analysis_prompt_gen,
                analysis_output_key,
                update_current_thought=False, # This step doesn't update the main thought flow
                is_final_answer_step=False
            )
            # Parse the JSON response from the analysis persona
            analysis_summary = {}
            if raw_analysis_response:
                try:
                    # Attempt to parse the response as JSON.
                    # We don't use LLMOutputParser here as the output is not a CODE_CHANGES structure.
                    analysis_summary = json.loads(raw_analysis_response)
                    self._update_status("Codebase context analysis successful.")
                except json.JSONDecodeError:
                    self._update_status(f"[yellow]Warning: Failed to parse context analysis response as JSON. Using default summary.[/yellow]")
                    analysis_summary = self._get_default_analysis_summary()
                except Exception as e:
                    self._update_status(f"[yellow]Warning: Unexpected error parsing context analysis response: {e}. Using default summary.[/yellow]")
                    analysis_summary = self._get_default_analysis_summary()
            else:
                self._update_status("[yellow]Context analysis persona returned empty response. Using default summary.[/yellow]")
                analysis_summary = self._get_default_analysis_summary()
            # REF_005: Validate the analysis summary structure
            if not self._validate_analysis_summary(analysis_summary):
                self._update_status("[yellow]Codebase analysis summary validation failed. Using default context.[/yellow]")
                return self._get_default_analysis_summary()
            return analysis_summary
        except (TokenBudgetExceededError, LLMProviderError, ValueError, RuntimeError, Exception) as e:
            # Catch errors during the analysis persona step
            error_message = f"[ERROR] Context analysis failed: {e}"
            self.intermediate_steps[f"{analysis_output_key}_Error"] = error_message
            self._update_status(error_message, state="error")
            return self._get_default_analysis_summary()
    def _execute_persona_step(self, persona_name: str, step_prompt_generator: Callable[[], str], output_key: str, max_retries_on_fail: int = 1, update_current_thought: bool = False, is_final_answer_step: bool = False) -> str:
        """Executes a single persona step, handling token budget, status updates, and parsing/validation errors."""
        persona = self._get_persona(persona_name)
        step_prompt = step_prompt_generator()
        estimated_input_tokens = self.gemini_provider.count_tokens(prompt=step_prompt, system_prompt=persona.system_prompt)
        remaining_budget = self.max_total_tokens_budget - self.cumulative_token_usage
        max_output_for_request = max(0, min(persona.max_tokens, remaining_budget - estimated_input_tokens))
        if estimated_input_tokens >= remaining_budget:
             raise TokenBudgetExceededError(f"Prompt for '{persona_name}' ({estimated_input_tokens} tokens) exceeds remaining budget ({remaining_budget} tokens).")
        for attempt in range(max_retries_on_fail + 1):
            current_persona_name = persona_name
            current_persona = persona
            current_step_prompt = step_prompt
            current_output_key = output_key
            if attempt > 0:
                current_persona_name = "Generalist_Assistant"
                if "Generalist_Assistant" not in self.all_personas:
                    self._update_status(f"[red]Error: Generalist_Assistant not found for fallback. Aborting.[/red]", state="error")
                    raise LLMUnexpectedError("Generalist_Assistant persona not found for fallback.")
                current_persona = self._get_persona(current_persona_name)
                current_step_prompt = (f"The previous attempt to process the following prompt with persona '{persona_name}' failed. "
                                        f"Please provide a general, concise summary or attempt to answer the original prompt given the context. "
                                        f"Original prompt:\n{step_prompt}")
                current_output_key = f"{output_key}_Fallback_Attempt_{attempt}"
                self._update_status(f"[yellow]Warning: Persona '{persona_name}' failed. Attempting fallback to '{current_persona_name}' (Attempt {attempt}/{max_retries_on_fail}).[/yellow]", state="running")
            estimated_next_step_cost = self.gemini_provider.calculate_usd_cost(estimated_input_tokens, max_output_for_request)
            self._update_status(f"Running persona: {current_persona_name} (Input: {estimated_input_tokens} tokens, Output: {max_output_for_request} tokens)...",
                                current_total_tokens=self.cumulative_token_usage,
                                current_total_cost=self.cumulative_usd_cost,
                                estimated_next_step_tokens=estimated_input_tokens + max_output_for_request,
                                estimated_next_step_cost=estimated_next_step_cost)
            try:
                raw_response_text, input_tokens, output_tokens = self.gemini_provider.generate(
                    prompt=current_step_prompt,
                    system_prompt=current_persona.system_prompt,
                    temperature=current_persona.temperature,
                    max_tokens=max_output_for_request
                )
                tokens_used = input_tokens + output_tokens
                cost_this_step = self.gemini_provider.calculate_usd_cost(input_tokens, output_tokens)
                parsed_data = None
                if raw_response_text:
                    try:
                        # Use the LLMOutputParser to parse and validate the raw response.
                        # This method is expected to always return a dictionary, even on failure,
                        # containing 'malformed_blocks' and 'code_changes' keys.
                        parsed_data = self.parser.parse_and_validate(raw_response_text)
                        if parsed_data.get("malformed_blocks"): # Check if malformed_blocks is not empty
                            malformed_blocks = parsed_data.get("malformed_blocks")
                            error_message = f"LLM output parsing/validation failed for persona '{current_persona_name}'.\n" + "\n".join(malformed_blocks)
                            self.intermediate_steps[f"{current_output_key}_Error"] = error_message
                            self._update_status(error_message, state="error")
                            # If parsing/validation fails, we cannot proceed with storing valid data.
                            # Raise an error to be caught by the outer handler.
                            raise LLMProviderError(f"Failed to parse/validate LLM output from {current_persona_name}.")
                        # Store the parsed and validated data (even if it has validation issues)
                        self.intermediate_steps[current_output_key] = parsed_data
                    except (LLMProviderError, ValueError, RuntimeError) as parse_err:
                        # Catch errors from parser or validation steps
                        error_message = f"[ERROR] Processing LLM output for '{current_persona_name}' failed: {parse_err}"
                        self.intermediate_steps[f"{current_output_key}_Error"] = error_message
                        self._update_status(error_message, state="error")
                        if attempt == max_retries_on_fail:
                            raise # Re-raise if max retries reached
                        continue # Try next attempt
                    except Exception as e:
                        # Catch any other unexpected errors during parsing/validation
                        error_message = f"[ERROR] Unexpected error processing LLM output for '{current_persona_name}': {e}"
                        self.intermediate_steps[f"{current_output_key}_Error"] = error_message
                        self._update_status(error_message, state="error")
                        if attempt == max_retries_on_fail:
                            raise # Re-raise if max retries reached
                        continue # Try next attempt
                else:
                    # Handle cases where raw_response_text is empty or None
                    self.intermediate_steps[current_output_key] = "[INFO] LLM returned empty response."
                    tokens_used = input_tokens # Only input tokens were used
                    cost_this_step = self.gemini_provider.calculate_usd_cost(input_tokens, 0)
                self.cumulative_token_usage += tokens_used # Update cumulative tokens
                self.cumulative_usd_cost += cost_this_step # Update cumulative cost
                if update_current_thought: self.current_thought = raw_response_text # Use raw response for thought if not parsed
                if is_final_answer_step: self.final_answer = raw_response_text # Store raw response as final answer if it's the last step
                self._update_status(f"{current_persona_name} completed. Used {tokens_used} tokens.",
                                    current_total_tokens=self.cumulative_token_usage,
                                    current_total_cost=self.cumulative_usd_cost)
                return raw_response_text # Return raw response if parsing failed or was not the final step
            except LLMProviderError as e:
                error_msg = f"[ERROR] Persona '{current_persona_name}' failed: {e}"
                self.intermediate_steps[current_output_key] = error_msg
                self._update_status(error_msg, state="error")
                if attempt == max_retries_on_fail:
                    raise
            except Exception as e:
                error_msg = f"[ERROR] Unexpected error during '{current_persona_name}' execution: {e}"
                self.intermediate_steps[current_output_key] = error_msg
                self._update_status(error_msg, state="error")
                if attempt == max_retries_on_fail:
                    raise
        raise LLMUnexpectedError(f"Max retries exceeded for persona '{persona_name}'.")
    def run_debate(self, max_turns: int = 5) -> Tuple[str, Dict[str, Any]]:
        """Executes the full Socratic debate loop."""
        if not self.personas:
            return {"error": "No personas loaded for the selected domain. Cannot run debate."}, {}
        self._update_status("Starting Socratic Arbitration Loop...",
                            current_total_tokens=self.cumulative_token_usage,
                            current_total_cost=self.cumulative_usd_cost)
        context_string = self.prepare_context()
        codebase_analysis_summary = self._analyze_codebase_context()
        analysis_str = json.dumps(codebase_analysis_summary, indent=2)
        def visionary_prompt_gen():
            if codebase_analysis_summary:
                return (f"USER PROMPT: {self.initial_prompt}\n\nCODEBASE CONTEXT ANALYSIS:\n{analysis_str}\n\n"
                        f"CODEBASE CONTEXT:\n{context_string}\n\nINSTRUCTIONS:\n"
                        f"1. **Analyze the provided codebase context and its analysis thoroughly.** Understand its structure, style, patterns, dependencies, and overall logic, incorporating insights from the analysis.\n"
                        f"2. **Propose an initial implementation strategy or code snippet.** Your proposal should be consistent with the existing codebase and address any identified concerns from the analysis.\n"
                        f"3. **Ensure your proposed code fits naturally into the existing architecture and follows its conventions.** Use the provided `GeminiProvider` and `SocraticDebate` classes as examples of how to integrate.")
            return (f"USER PROMPT: {self.initial_prompt}\n\n"
                    f"CODEBASE CONTEXT:\n{context_string}\n\n"
                    f"INSTRUCTIONS:\n"
                    f"1. **Analyze the provided codebase context thoroughly.** Understand its structure, style, patterns, dependencies, and overall logic.\n"
                    f"2. **Propose an initial implementation strategy or code snippet.** Your proposal should be consistent with the existing codebase.\n"
                    f"3. **Ensure your proposed code fits naturally into the existing architecture and follows its conventions.** Use the provided `GeminiProvider` and `SocraticDebate` classes as examples of how to integrate.")
        try:
            visionary_output = self._execute_persona_step("Visionary_Generator", visionary_prompt_gen, "Visionary_Generator_Output", update_current_thought=True)
            def skeptical_prompt_gen():
                return f"Critique the following proposal from a highly skeptical, risk-averse perspective. Identify potential failure points, architectural flaws, or critical vulnerabilities:\n\n{visionary_output}"
            skeptical_critique = self._execute_persona_step("Skeptical_Generator", skeptical_prompt_gen, "Skeptical_Critique")
            domain_critiques_text = ""
            for persona_name_in_sequence in self.persona_sequence:
                if persona_name_in_sequence in self.personas:
                    if persona_name_in_sequence not in ["Visionary_Generator", "Skeptical_Generator", "Constructive_Critic", "Impartial_Arbitrator", "Devils_Advocate", "Generalist_Assistant"]:
                        def expert_prompt_gen(name=persona_name_in_sequence, proposal=visionary_output):
                            return (
                                f"As a {name.replace('_', ' ')}, analyze the following proposal from your expert perspective. "
                                f"Identify specific points of concern, risks, or areas for improvement relevant to your domain. "
                                f"Your insights will be crucial for subsequent synthesis and refinement steps, so be thorough and specific. "
                                f"Present your analysis in a structured format, using clear headings or bullet points for 'Concerns' and 'Recommendations'.\n\n"
                                f"Proposal:\n{proposal}")
                        critique = self._execute_persona_step(persona_name_in_sequence, expert_prompt_gen, f"{persona_name_in_sequence}_Critique")
                        domain_critiques_text += f"\n\n--- {persona_name_in_sequence.replace('_', ' ')} Critique ---\n{critique}"
            def constructive_prompt_gen():
                return (
                    f"Original Proposal:\n{visionary_output}\n\n"
                    f"--- Skeptical Critique ---\n{skeptical_critique}\n"
                    f"{domain_critiques_text}\n\n"
                    f"Based on all the above inputs, provide specific, actionable improvements. Synthesize the critiques, resolve conflicts where possible, and propose a refined solution or code.")
            constructive_feedback = self._execute_persona_step("Constructive_Critic", constructive_prompt_gen, "Constructive_Critic_Output")
            def arbitrator_prompt_gen():
                return (
                    f"Synthesize all the following information into a single, balanced, and definitive final answer. Your output MUST be a JSON object with the following structure:\n\n```json\n{{\n  \"COMMIT_MESSAGE\": \"<string>\",\n  \"RATIONALE\": \"<string, including CONFLICT RESOLUTION: or UNRESOLVED CONFLICT: if applicable>\"\n  \"CODE_CHANGES\": [\n    {{\n      \"file_path\": \"<string>\",\n      \"action\": \"ADD | MODIFY | REMOVE\",\n      \"full_content\": \"<string>\" (Required for ADD/MODIFY actions, REPRESENTING THE ENTIRE NEW FILE CONTENT OR MODIFIED FILE CONTENT. ENSURE ALL DOUBLE QUOTES WITHIN THE CONTENT ARE ESCAPED AS \\\".)\n    }},\n    {{\n      \"file_path\": \"<string>\",\n      \"action\": \"REMOVE\",\n      \"lines\": [\"<string>\", \"<string>\"] (Required for REMOVE action, representing the specific lines to be removed)\n    }}\n  ]\n}}\n```\n\nEnsure that the `CODE_CHANGES` array contains objects for each file change. For `MODIFY` and `ADD` actions, provide the `full_content` of the file. For `REMOVE` actions, provide an array of `lines` to be removed. If there are conflicting suggestions, you must identify them and explain your resolution in the 'RATIONALE' section, starting with 'CONFLICT RESOLUTION: '.\nIf a conflict cannot be definitively resolved or requires further human input, flag it clearly in the 'RATIONALE' starting with 'UNRESOLVED CONFLICT: '.\n*   **Code Snippets:** Ensure all code snippets within `full_content` are correctly formatted and escaped, especially double quotes.\n*   **Clarity and Conciseness:** Present the final plan clearly and concisely.\n*   **Unit Tests:** For any `ADD` or `MODIFY` action in `CODE_CHANGES`, if the file is a Python file, you MUST also propose a corresponding unit test file (e.g., `tests/test_new_module.py` or `tests/test_modified_function.py`) in a separate `CODE_CHANGES` entry with action `ADD` and its `full_content`. Ensure these tests are comprehensive and follow standard Python `unittest` or `pytest` practices.\n\n--- DEBATE SUMMARY ---\n"
                     f"User Prompt: {self.initial_prompt}\n\n"
                     f"Visionary Proposal:\n{visionary_output}\n\n"
                     f"Skeptical Critique:\n{skeptical_critique}\n"
                     f"{domain_critiques_text}\n\n"
                     f"Constructive Feedback:\n{constructive_feedback}\n\n"
                     f"--- END DEBATE ---"
                 )
            arbitrator_raw_response = self._execute_persona_step("Impartial_Arbitrator", arbitrator_prompt_gen, "Impartial_Arbitrator_Output", is_final_answer_step=True)
            def devil_prompt_gen():
                return f"Critique the following final synthesized answer (which will be a JSON object). Find the single most critical, fundamental flaw. Do not offer solutions, only expose the weakness with a sharp, incisive critique. Focus on non-obvious issues like race conditions, scalability limits, or subtle security holes:\n{self.final_answer}"
            self._execute_persona_step("Devils_Advocate", devil_prompt_gen, "Devils_Advocate_Critique")
            self.intermediate_steps["Total_Tokens_Used"] = self.cumulative_token_usage
            self.intermediate_steps["Total_Estimated_Cost_USD"] = self.cumulative_usd_cost
            self._update_status(f"Socratic Arbitration Loop finished. Total tokens used: {self.cumulative_token_usage:,}. Total estimated cost: ${self.cumulative_usd_cost:.4f}",
                                state="complete", expanded=False,
                                current_total_tokens=self.cumulative_token_usage,
                                current_total_cost=self.cumulative_usd_cost)
            return self.final_answer, self.intermediate_steps
        except (TokenBudgetExceededError, LLMProviderError, ValueError, RuntimeError, Exception) as e:
            # Catch specific errors related to LLM calls, parsing, validation, or general exceptions
            error_msg = f"[ERROR] Socratic Debate failed: {e}"
            self.intermediate_steps["Debate_Error"] = error_msg
            self._update_status(error_msg, state="error")
            # Ensure final_answer reflects the error
            self.final_answer = f"Socratic Debate failed: {e}"
            # Update metrics even on error if possible
            self.intermediate_steps["Total_Tokens_Used"] = self.cumulative_token_usage
            self.intermediate_steps["Total_Estimated_Cost_USD"] = self.cumulative_usd_cost
            raise # Re-raise the exception to be caught by the caller (e.g., Streamlit app)
    def _validate_analysis_summary(self, summary: Dict[str, Any]) -> bool:
        """REF_005: Validate the structure and content of the analysis summary."""
        required_keys = ["key_modules", "security_concerns", "architectural_patterns", "performance_bottlenecks"]
        if not isinstance(summary, dict):
            return False
        if not all(key in summary for key in required_keys):
            return False
        if not isinstance(summary["key_modules"], list) or \
           not isinstance(summary["security_concerns"], list) or \
           not isinstance(summary["architectural_patterns"], list) or \
           not isinstance(summary["performance_bottlenecks"], list):
            return False
        return True
    def _get_default_analysis_summary(self) -> Dict[str, Any]:
        """Provides a default summary if validation fails or analysis is skipped."""
        return {
            "key_modules": [],
            "security_concerns": [],
            "architectural_patterns": [],
            "performance_bottlenecks": []
        }