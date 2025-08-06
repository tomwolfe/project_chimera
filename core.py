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
import logging
from rich.console import Console
from pydantic import BaseModel, Field, ValidationError, model_validator, validator
import streamlit as st
from typing import List, Dict, Tuple, Any, Callable, Optional
from llm_provider import GeminiProvider, LLMProviderError, GeminiAPIError, LLMUnexpectedError
from src.utils.output_parser import LLMOutputParser
from src.utils.code_validator import validate_code_output_batch
from src.models import PersonaConfig, ReasoningFrameworkConfig

class TokenBudgetExceededError(LLMProviderError):
    """Raised when an LLM call would exceed the total token budget."""
    pass

@st.cache_resource
def load_personas(file_path: str = 'personas.yaml') -> Tuple[Dict[str, PersonaConfig], Dict[str, List[str]], List[str], str]:
    """Loads persona configurations from a YAML file. Cached using st.cache_resource."""
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
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
            # MODIFIED: Removed _status_callback from GeminiProvider init
            self.gemini_provider = GeminiProvider(api_key=api_key, model_name=model_name)
        self.parser = LLMOutputParser(self.gemini_provider)
        self.cumulative_token_usage = 0
        self.cumulative_usd_cost = 0.0
        self.intermediate_steps: Dict[str, Any] = {}
        self.rich_console = rich_console if rich_console else Console()
        self.current_thought = initial_prompt
        self.final_answer = "Process did not complete."
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

    @staticmethod
    @st.cache_data(ttl=3600)
    def _prioritize_python_code(_gemini_provider: GeminiProvider, content: str, max_tokens: int) -> str:
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
            logging.warning(f"[yellow]Warning: Syntax error in Python context file, falling back to simple truncation.[/yellow]")
            # MODIFIED: Pass _status_callback=None for static method
            return SocraticDebate._truncate_text_by_tokens(_gemini_provider, content, max_tokens, _status_callback=None)
        combined_content = "\n".join(priority_lines + other_lines)
        # MODIFIED: Pass _status_callback=None for static method
        return SocraticDebate._truncate_text_by_tokens(_gemini_provider, combined_content, max_tokens, _status_callback=None)

    @staticmethod
    def _truncate_text_by_tokens(_gemini_provider: GeminiProvider, text: str, max_tokens: int, _status_callback: Callable = None) -> str:
        """Truncates text to fit within max_tokens using the GeminiProvider's token counting."""
        if not text:
            return ""
        # MODIFIED: Pass _status_callback to count_tokens
        current_tokens = _gemini_provider.count_tokens(text, "", _status_callback=_status_callback)
        if current_tokens <= max_tokens:
            return text
        chars_per_token_estimate = 4
        target_chars = max_tokens * chars_per_token_estimate
        truncated_text = text
        if len(truncated_text) > target_chars:
            truncated_text = truncated_text[:target_chars]
        # MODIFIED: Pass _status_callback to count_tokens
        while _gemini_provider.count_tokens(truncated_text, "", _status_callback=_status_callback) > max_tokens and len(truncated_text) > 0:
            chars_to_remove = max(1, len(truncated_text) // 20)
            truncated_text = truncated_text[:-chars_to_remove]
            if len(truncated_text) == 0:
                break
        # MODIFIED: Pass _status_callback to count_tokens
        if _gemini_provider.count_tokens(text, "", _status_callback=_status_callback) > max_tokens:
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
            # MODIFIED: Pass status_callback to count_tokens
            header_tokens = self.gemini_provider.count_tokens(header, "", _status_callback=self.status_callback)
            remaining_budget_for_file_content = context_budget - current_tokens - header_tokens
            if remaining_budget_for_file_content <= 0:
                self._update_status(f"Skipping file '{path}' due to context token budget exhaustion.")
                break
            file_content_to_add = ""
            if path.endswith('.py'):
                # MODIFIED: Pass the gemini_provider instance and status_callback to _prioritize_python_code
                prioritized_content = SocraticDebate._prioritize_python_code(self.gemini_provider, content, remaining_budget_for_file_content)
                # MODIFIED: Pass self.gemini_provider and status_callback to _truncate_text_by_tokens
                file_content_to_add = SocraticDebate._truncate_text_by_tokens(self.gemini_provider, prioritized_content, remaining_budget_for_file_content, _status_callback=self.status_callback)
            else:
                # MODIFIED: Pass self.gemini_provider and status_callback to _truncate_text_by_tokens
                file_content_to_add = SocraticDebate._truncate_text_by_tokens(self.gemini_provider, content, remaining_budget_for_file_content, _status_callback=self.status_callback)
            if not file_content_to_add:
                continue
            full_file_block = header + file_content_to_add + "\n"
            # MODIFIED: Pass status_callback to count_tokens
            file_block_tokens = self.gemini_provider.count_tokens(full_file_block, "", _status_callback=self.status_callback)
            if current_tokens + file_block_tokens > context_budget:
                self._update_status(f"Warning: Could not fit full file '{path}' even after truncation. Skipping remaining files.")
                break
            context_str_parts.append(full_file_block)
            current_tokens += file_block_tokens
        # MODIFIED: Pass status_callback to count_tokens
        self._update_status(f"Prepared codebase context using {self.gemini_provider.count_tokens(final_context_string, '', _status_callback=self.status_callback)} tokens.")
        return final_context_string

    def _analyze_codebase_context(self) -> Dict[str, Any]:
        """
        Analyzes the codebase context using a dedicated persona.
        CRE_001: Introduce a 'Context Analysis' step using a dedicated persona.
        """
        if not self.codebase_context:
            self._update_status("[yellow]No codebase context provided for analysis.[/yellow]")
            return self._get_default_analysis_summary()
        if "Context_Aware_Assistant" not in self.personas and "Context_Aware_Assistant" not in self.all_personas:
            self._update_status("[yellow]Context_Aware_Assistant persona not found. Skipping context analysis.[/yellow]")
            return self._get_default_analysis_summary()
        self._update_status("Analyzing codebase context with Context_Aware_Assistant...")
        context_string_for_analysis = self.prepare_context()
        
        def analysis_prompt_gen():
            return (f"CODEBASE CONTEXT:\n{context_string_for_analysis}\n\n"
                    f"INSTRUCTIONS:\n"
                    f"Analyze the provided codebase context thoroughly. Understand its structure, style, patterns, dependencies, and overall logic.\n"
                    f"Provide a concise summary of the codebase in the specified JSON format.")

        analysis_output_key = "Context_Analysis_Output"
        try:
            # MODIFIED: Pass the function itself, not its call
            raw_analysis_response = self._execute_persona_step(
                "Context_Aware_Assistant",
                analysis_prompt_gen,
                analysis_output_key,
                update_current_thought=False,
                is_final_answer_step=False
            )
            analysis_summary = {}
            if raw_analysis_response:
                try:
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
            if not self._validate_analysis_summary(analysis_summary):
                self._update_status("[yellow]Codebase analysis summary validation failed. Using default context.[/yellow]")
                return self._get_default_analysis_summary()
            return analysis_summary
        except (TokenBudgetExceededError, LLMProviderError, ValueError, RuntimeError, Exception) as e:
            error_message = f"[ERROR] Context analysis failed: {e}"
            self.intermediate_steps[f"{analysis_output_key}_Error"] = error_message
            self._update_status(error_message, state="error")
            return self._get_default_analysis_summary()

    # MODIFIED: Changed return type to Any to accommodate both str and Dict
    def _execute_persona_step(self, persona_name: str, step_prompt_generator: Callable[[], str], output_key: str, max_retries_on_fail: int = 1, update_current_thought: bool = False, is_final_answer_step: bool = False) -> Any:
        """Executes a single persona step, handling token budget, status updates, and parsing/validation errors."""
        persona = self._get_persona(persona_name)
        step_prompt = step_prompt_generator() # MODIFIED: Call the prompt generator function
        
        # MODIFIED: Pass status_callback to count_tokens
        estimated_input_tokens = self.gemini_provider.count_tokens(prompt=step_prompt, system_prompt=persona.system_prompt, _status_callback=self.status_callback)
        remaining_budget = self.max_total_tokens_budget - self.cumulative_token_usage
        max_output_for_request = max(0, min(persona.max_tokens, remaining_budget - estimated_input_tokens))
        if estimated_input_tokens >= remaining_budget:
             raise TokenBudgetExceededError(f"Prompt for '{persona_name}' ({estimated_input_tokens} tokens) exceeds remaining budget ({remaining_budget} tokens).")
        for attempt in range(max_retries_on_fail + 1):
            current_persona_name = persona_name
            current_persona = persona
            current_step_prompt = step_prompt
            current_output_key = output_key
            
            # MODIFIED: Special handling for Impartial_Arbitrator fallback
            if attempt > 0:
                current_persona_name = "Generalist_Assistant"
                if "Generalist_Assistant" not in self.all_personas:
                    self._update_status(f"[red]Error: Generalist_Assistant not found for fallback. Aborting.[/red]", state="error")
                    raise LLMUnexpectedError("Generalist_Assistant persona not found for fallback.")
                current_persona = self._get_persona(current_persona_name)
                
                if is_final_answer_step and persona_name == "Impartial_Arbitrator":
                    # If Arbitrator fails, Generalist_Assistant provides a plain text summary
                    current_step_prompt = (
                        f"The previous attempt to synthesize the final answer failed. "
                        f"Please provide a concise, plain-text summary of the debate and the proposed solution. "
                        f"Do NOT attempt to generate JSON. Focus on the key takeaways and any unresolved issues. "
                        f"Original prompt for final answer:\n{step_prompt}"
                    )
                else:
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
                # MODIFIED: Pass status_callback to generate
                raw_response_text, input_tokens, output_tokens = self.gemini_provider.generate(
                    prompt=current_step_prompt,
                    system_prompt=current_persona.system_prompt,
                    temperature=current_persona.temperature,
                    max_tokens=max_output_for_request,
                    _status_callback=self.status_callback
                )
                tokens_used = input_tokens + output_tokens
                cost_this_step = self.gemini_provider.calculate_usd_cost(input_tokens, output_tokens)
                
                # MODIFIED: Handle parsing and return based on is_final_answer_step
                if is_final_answer_step:
                    parsed_data = None
                    if raw_response_text:
                        try:
                            parsed_data = self.parser.parse_and_validate(raw_response_text)
                            if parsed_data.get("malformed_blocks"):
                                error_message = f"LLM output parsing/validation failed for persona '{current_persona_name}'. See malformed_blocks for details."
                                self.intermediate_steps[f"{current_output_key}_Error"] = error_message
                                self._update_status(error_message, state="error")
                        except Exception as parse_err:
                            error_message = f"[ERROR] Unexpected error processing LLM output for '{current_persona_name}': {parse_err}"
                            self.intermediate_steps[f"{current_output_key}_Error"] = error_message
                            self._update_status(error_message, state="error")
                            parsed_data = {
                                "commit_message": "Parsing error",
                                "rationale": f"Failed to parse LLM output as JSON. Error: {parse_err}\nRaw output: {raw_response_text[:500]}...",
                                "code_changes": [],
                                "conflict_resolution": None,
                                "unresolved_conflict": None,
                                "malformed_blocks": [f"Unexpected error during parsing: {parse_err}", f"Raw output:\n{raw_response_text}"]
                            }
                    else:
                        parsed_data = {
                            "commit_message": "Empty response",
                            "rationale": "[INFO] LLM returned empty response for final answer.",
                            "code_changes": [],
                            "conflict_resolution": None,
                            "unresolved_conflict": None,
                            "malformed_blocks": ["LLM returned empty response for final answer."]
                        }
                        tokens_used = input_tokens
                        cost_this_step = self.gemini_provider.calculate_usd_cost(input_tokens, 0)
                    
                    self.intermediate_steps[current_output_key] = parsed_data
                    self.cumulative_token_usage += tokens_used
                    self.cumulative_usd_cost += cost_this_step
                    self.final_answer = parsed_data
                    self._update_status(f"{current_persona_name} completed. Used {tokens_used} tokens.",
                                        current_total_tokens=self.cumulative_token_usage,
                                        current_total_cost=self.cumulative_usd_cost)
                    return parsed_data
                else: # Not a final answer step, return raw text
                    if raw_response_text:
                        self.intermediate_steps[current_output_key] = raw_response_text
                    else:
                        self.intermediate_steps[current_output_key] = "[INFO] LLM returned empty response."
                        tokens_used = input_tokens
                        cost_this_step = self.gemini_provider.calculate_usd_cost(input_tokens, 0)
                    
                    self.cumulative_token_usage += tokens_used
                    self.cumulative_usd_cost += cost_this_step
                    if update_current_thought: self.current_thought = raw_response_text
                    self._update_status(f"{current_persona_name} completed. Used {tokens_used} tokens.",
                                        current_total_tokens=self.cumulative_token_usage,
                                        current_total_cost=self.cumulative_usd_cost)
                    return raw_response_text
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

    def run_debate(self, max_turns: int = 5) -> Tuple[Any, Dict[str, Any]]:
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
                    if persona_name_in_sequence not in ["Visionary_Generator", "Skeptical_Generator", "Constructive_Critic", "Impartial_Arbitrator", "Devils_Advocate", "Generalist_Assistant", "Context_Aware_Assistant"]:
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
                    f"Synthesize all the following information into a single, balanced, and definitive final answer. Your output MUST be a JSON object with the following structure:\n\n```json\n{{\n  \"COMMIT_MESSAGE\": \"<string>\",\n  \"RATIONALE\": \"<string, including CONFLICT_RESOLUTION: or UNRESOLVED_CONFLICT: if applicable>\",\n  \"CODE_CHANGES\": [\n    {{\n      \"FILE_PATH\": \"<string>\",\n      \"ACTION\": \"ADD | MODIFY | REMOVE\",\n      \"FULL_CONTENT\": \"<string>\" (Required for ADD/MODIFY actions, REPRESENTING THE ENTIRE NEW FILE CONTENT OR MODIFIED FILE CONTENT. ENSURE ALL DOUBLE QUOTES WITHIN THE CONTENT ARE ESCAPED AS \\\".)\n    }},\n    {{\n      \"FILE_PATH\": \"<string>\",\n      \"ACTION\": \"REMOVE\",\n      \"LINES\": [\"<string>\", \"<string>\"] (Required for REMOVE action, representing the specific lines to be removed)\n    }}\n  ]\n}}\n```\n\nEnsure that the `CODE_CHANGES` array contains objects for each file change. For `MODIFY` and `ADD` actions, provide the `FULL_CONTENT` of the file. For `REMOVE` actions, provide an array of `LINES` to be removed. If there are conflicting suggestions, you must identify them and explain your resolution in the 'RATIONALE' section, starting with 'CONFLICT_RESOLUTION: '.\nIf a conflict cannot be definitively resolved or requires further human input, flag it clearly in the 'RATIONALE' starting with 'UNRESOLVED_CONFLICT: '.\n*   **Code Snippets:** Ensure all code snippets within `FULL_CONTENT` are correctly formatted and escaped, especially double quotes.\n*   **Clarity and Conciseness:** Present the final plan clearly and concisely.\n*   **Unit Tests:** For any `ADD` or `MODIFY` action in `CODE_CHANGES`, if the file is a Python file, you MUST also propose a corresponding unit test file (e.g., `tests/test_new_module.py` or `tests/test_modified_function.py`) in a separate `CODE_CHANGES` entry with action `ADD` and its `FULL_CONTENT`. Ensure these tests are comprehensive and follow standard Python `unittest` or `pytest` practices.\n\n--- DEBATE SUMMARY ---\n"
                     f"User Prompt: {self.initial_prompt}\n\n"
                     f"Visionary Proposal:\n{visionary_output}\n\n"
                     f"Skeptical Critique:\n{skeptical_critique}\n"
                     f"{domain_critiques_text}\n\n"
                     f"Constructive Feedback:\n{constructive_feedback}\n\n"
                     f"--- END DEBATE ---"
                 )
            # MODIFIED: arbitrator_output_dict will now be a dictionary (parsed_data)
            arbitrator_output_dict = self._execute_persona_step("Impartial_Arbitrator", arbitrator_prompt_gen, "Impartial_Arbitrator_Output", is_final_answer_step=True)
            
            # MODIFIED: Use the parsed_data for Devil's Advocate prompt
            devil_advocate_input = json.dumps(arbitrator_output_dict, indent=2) if isinstance(arbitrator_output_dict, dict) else str(arbitrator_output_dict)

            def devil_prompt_gen():
                return f"Critique the following final synthesized answer (which will be a JSON object). Find the single most critical, fundamental flaw. Do not offer solutions, only expose the weakness with a sharp, incisive critique. Focus on non-obvious issues like race conditions, scalability limits, or subtle security holes:\n{devil_advocate_input}"
            
            self._execute_persona_step("Devils_Advocate", devil_prompt_gen, "Devils_Advocate_Critique")
            
            self.intermediate_steps["Total_Tokens_Used"] = self.cumulative_token_usage
            self.intermediate_steps["Total_Estimated_Cost_USD"] = self.cumulative_usd_cost
            self._update_status(f"Socratic Arbitration Loop finished. Total tokens used: {self.cumulative_token_usage:,}. Total estimated cost: ${self.cumulative_usd_cost:.4f}",
                                state="complete", expanded=False,
                                current_total_tokens=self.cumulative_token_usage,
                                current_total_cost=self.cumulative_usd_cost)
            # MODIFIED: Return the parsed dictionary for the final answer
            return self.final_answer, self.intermediate_steps
        except (TokenBudgetExceededError, LLMProviderError, ValueError, RuntimeError, Exception) as e:
            error_msg = f"[ERROR] Socratic Debate failed: {e}"
            self.intermediate_steps["Debate_Error"] = error_msg
            self._update_status(error_msg, state="error")
            # MODIFIED: final_answer is now a dict, so populate it with error info
            self.final_answer = {
                "commit_message": "Debate Failed",
                "rationale": f"Socratic Debate failed: {e}",
                "code_changes": [],
                "conflict_resolution": None,
                "unresolved_conflict": None,
                "malformed_blocks": [f"Debate failed: {e}"]
            }
            self.intermediate_steps["Total_Tokens_Used"] = self.cumulative_token_usage
            self.intermediate_steps["Total_Estimated_Cost_USD"] = self.cumulative_usd_cost
            raise

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