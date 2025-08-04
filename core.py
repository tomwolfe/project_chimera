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
from pydantic import BaseModel, Field, ValidationError, model_validator
import streamlit as st
from typing import List, Dict, Tuple, Any, Callable, Optional
from llm_provider import GeminiProvider, LLMProviderError, GeminiAPIError, LLMUnexpectedError

# --- Custom Exception for Token Budget ---
class TokenBudgetExceededError(LLMProviderError):
    """Raised when an LLM call would exceed the total token budget."""
    pass

# --- Pydantic Models for Persona Configuration ---
class Persona(BaseModel):
    name: str
    system_prompt: str
    temperature: float = Field(..., ge=0.0, le=1.0)
    max_tokens: int = Field(..., gt=0)
    description: Optional[str] = None

class FullPersonaConfig(BaseModel):
    personas: List[Persona] = Field(default_factory=list)
    persona_sets: Dict[str, List[str]] = Field(default_factory=lambda: {"General": []})

    @model_validator(mode='after')
    def validate_persona_sets_references(self):
        all_persona_names = {p.name for p in self.personas}
        for set_name, persona_names_in_set in self.persona_sets.items():
            if not isinstance(persona_names_in_set, list):
                raise ValueError(f"Persona set '{set_name}' must be a list of persona names.")
            for p_name in persona_names_in_set:
                if p_name not in all_persona_names:
                    raise ValueError(f"Persona '{p_name}' referenced in set '{set_name}' not found in 'personas' list.")
        return self

    @property
    def default_persona_set(self) -> str:
        return "General" if "General" in self.persona_sets else next(iter(self.persona_sets.keys()))

@st.cache_resource
def load_personas(file_path: str = "personas.yaml") -> Tuple[Dict[str, Persona], Dict[str, List[str]], str]:
    """Loads persona configurations from a YAML file. Cached using st.cache_resource."""
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        full_config = FullPersonaConfig(**data)
        all_personas_dict = {p.name: p for p in full_config.personas}
        return all_personas_dict, full_config.persona_sets, full_config.default_persona_set
    except (FileNotFoundError, ValidationError, yaml.YAMLError) as e:
        print(f"Error loading personas from {file_path}: {e}")
        raise

class SocraticDebate:
    DEFAULT_MAX_RETRIES = 2
    MAX_BACKOFF_SECONDS = 30
    # CONTEXT_TOKEN_BUDGET_RATIO is now passed as an argument during initialization

    def __init__(self,
                 initial_prompt: str,
                 api_key: str,
                 max_total_tokens_budget: int,
                 model_name: str,
                 personas: Dict[str, Persona],
                 all_personas: Dict[str, Persona],
                 persona_sets: Dict[str, List[str]],
                 gemini_provider: Optional[GeminiProvider] = None,
                 domain: str = "General",
                 status_callback: Callable = None,
                 rich_console: Optional[Console] = None,
                 codebase_context: Optional[Dict[str, str]] = None,
                 context_token_budget_ratio: float = 0.25): # Add this parameter
        self.initial_prompt = initial_prompt
        self.max_total_tokens_budget = max_total_tokens_budget
        self.model_name = model_name
        self.personas = personas
        self.domain = domain
        self.all_personas = all_personas
        self.persona_sets = persona_sets
        self.status_callback = status_callback
        self.context_token_budget_ratio = context_token_budget_ratio # Store it
        
        if gemini_provider:
            self.gemini_provider = gemini_provider
        else:
            self.gemini_provider = GeminiProvider(api_key=api_key, model_name=model_name, status_callback=self._update_status)
            
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

    def _get_persona(self, name: str) -> Persona:
        """Retrieves a persona by name, checking both specific and all personas."""
        persona = self.personas.get(name) or self.all_personas.get(name)
        if not persona:
            raise ValueError(f"Persona '{name}' not found.")
        return persona

    @st.cache_data(ttl=3600) # Cache AST-based prioritization to speed up context preparation.
    def _prioritize_python_code(_self, content: str, max_tokens: int) -> str:
        """
        Prioritizes imports, class/function definitions for Python code.
        Truncates the content to fit within max_tokens.
        """
        lines = content.splitlines()
        priority_lines = []
        other_lines = []
        
        try:
            tree = ast.parse(content)
            # Collect all lines that are part of priority nodes
            priority_line_numbers = set()
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef)):
                    start_lineno = node.lineno - 1
                    # Use end_lineno if available, otherwise assume single line
                    end_lineno = node.end_lineno if hasattr(node, 'end_lineno') else start_lineno + 1
                    for i in range(start_lineno, end_lineno):
                        if i < len(lines):
                            priority_line_numbers.add(i)
            
            # Separate lines into priority and other
            for i, line in enumerate(lines):
                if i in priority_line_numbers:
                    priority_lines.append(line)
                else:
                    other_lines.append(line)

        except SyntaxError:
            # Fallback to simple line-by-line if AST parsing fails
            _self._update_status(f"[yellow]Warning: Syntax error in Python context file, falling back to simple truncation.[/yellow]")
            return _self._truncate_text_by_tokens(content, max_tokens)

        # Combine and truncate
        combined_content = "\n".join(priority_lines + other_lines)
        return _self._truncate_text_by_tokens(combined_content, max_tokens)

    def _truncate_text_by_tokens(self, text: str, max_tokens: int) -> str:
        """Truncates text to fit within max_tokens using the GeminiProvider's token counting."""
        if not text:
            return ""
        
        # Estimate tokens for the full text
        full_text_tokens = self.gemini_provider.count_tokens(text, "")
        if full_text_tokens <= max_tokens:
            return text

        # Simple iterative truncation from the end
        truncated_text = text
        # Calculate approximate chars per token (e.g., 4 chars/token)
        chars_per_token = 4 
        # Estimate how many characters to keep
        target_chars = max_tokens * chars_per_token
        
        if len(truncated_text) > target_chars:
            truncated_text = truncated_text[:target_chars]
        
        # Refine by token count, ensuring we don't get stuck in an infinite loop
        # and handle cases where even a single character might be too many tokens.
        while self.gemini_provider.count_tokens(truncated_text, "") > max_tokens and len(truncated_text) > 0:
            # Remove a portion of the text from the end. The amount to remove is heuristic; removing 5% or at least 1 char.
            chars_to_remove = max(1, len(truncated_text) // 20) 
            truncated_text = truncated_text[:-chars_to_remove]
            if len(truncated_text) == 0:
                break
        
        # Add an ellipsis if truncation actually occurred
        if self.gemini_provider.count_tokens(text, "") > max_tokens:
            return truncated_text.strip() + "\n... (truncated)"
        return truncated_text

    def prepare_context(self, codebase_files: Dict[str, str], debate_history: List[Dict[str, Any]]) -> str:
        """Prepares the codebase context, prioritizing Python code and truncating to fit budget."""
        if not self.codebase_context:
            return "No codebase context provided."

        context_budget = int(self.max_total_tokens_budget * self.context_token_budget_ratio)
        context_str_parts = []
        current_tokens = 0

        for path, content in self.codebase_context.items():
            header = f"--- file_path: {path} ---\n" # Added newline here
            header_tokens = self.gemini_provider.count_tokens(header, "")
            
            remaining_budget_for_file_content = context_budget - current_tokens - header_tokens
            if remaining_budget_for_file_content <= 0:
                self._update_status(f"Skipping file '{path}' due to context token budget.")
                break

            file_content_to_add = ""
            if path.endswith('.py'):
                # Use the AST-based prioritization for Python files
                prioritized_content = self._prioritize_python_code(content, remaining_budget_for_file_content)
                file_content_to_add = self._truncate_text_by_tokens(prioritized_content, remaining_budget_for_file_content)
            else:
                # Use the general truncation method for non-Python files
                file_content_to_add = self._truncate_text_by_tokens(content, remaining_budget_for_file_content)
            
            if not file_content_to_add:
                continue # Skip if content is empty after truncation

            full_file_block = header + file_content_to_add + "\n"
            file_block_tokens = self.gemini_provider.count_tokens(full_file_block, "")

            if current_tokens + file_block_tokens > context_budget:
                self._update_status(f"Warning: Could not fit full file '{path}' even after truncation. Skipping remaining files.")
                break # Stop adding more files

            context_str_parts.append(full_file_block)
            current_tokens += file_block_tokens

        final_context_string = "".join(context_str_parts)
        self._update_status(f"Prepared codebase context using {self.gemini_provider.count_tokens(final_context_string, '')} tokens.")
        return final_context_string

    def _execute_persona_step(self, persona_name: str, step_prompt_generator: Callable[[], str], output_key: str, max_retries_on_fail: int = 1, **kwargs) -> str:
        """Executes a single persona step, handling token budget and status updates."""
        persona = self._get_persona(persona_name)
        step_prompt = step_prompt_generator()
        
        estimated_input_tokens = self.gemini_provider.count_tokens(prompt=step_prompt, system_prompt=persona.system_prompt)
        remaining_budget = self.max_total_tokens_budget - self.cumulative_token_usage
        
        # Ensure max_output_for_request is not negative
        max_output_for_request = max(0, min(persona.max_tokens, remaining_budget - estimated_input_tokens))
        
        # Check if the estimated tokens for this step would exceed the budget
        if estimated_input_tokens + max_output_for_request < estimated_input_tokens: # Overflow check
             raise TokenBudgetExceededError(f"Estimated tokens for '{persona_name}' calculation overflowed.")
        if estimated_input_tokens >= remaining_budget:
             raise TokenBudgetExceededError(f"Prompt for '{persona_name}' ({estimated_input_tokens} tokens) exceeds remaining budget ({remaining_budget} tokens).")
        
        for attempt in range(max_retries_on_fail + 1): # +1 for the initial attempt
            current_persona_name = persona_name
            current_persona = persona
            current_step_prompt = step_prompt
            current_output_key = output_key

            if attempt > 0: # This is a retry, use fallback
                current_persona_name = "Generalist_Assistant"
                if "Generalist_Assistant" not in self.all_personas:
                    self._update_status(f"[red]Error: Generalist_Assistant not found for fallback. Aborting.[/red]", state="error")
                    raise ValueError("Generalist_Assistant persona not found for fallback.")
                current_persona = self._get_persona(current_persona_name)
                current_step_prompt = (f"The previous attempt to process the following prompt with persona '{persona_name}' failed. "
                                        f"Please provide a general, concise summary or attempt to answer the original prompt given the context. "
                                        f"Original prompt:\n{step_prompt}")
                current_output_key = f"{output_key}_Fallback_Attempt_{attempt}"
                self._update_status(f"[yellow]Warning: Persona '{persona_name}' failed. Attempting fallback to '{current_persona_name}' (Attempt {attempt}/{max_retries_on_fail}).[/yellow]", state="warning")

            self._update_status(f"Running persona: {current_persona_name}...",
                                current_total_tokens=self.cumulative_token_usage,
                                current_total_cost=self.cumulative_usd_cost,
                                estimated_next_step_tokens=estimated_input_tokens + max_output_for_request,
                                estimated_next_step_cost=self.gemini_provider.calculate_usd_cost(estimated_input_tokens, max_output_for_request))
            
            try:
                response_text, input_tokens, output_tokens = self.gemini_provider.generate(
                    prompt=current_step_prompt,
                    system_prompt=current_persona.system_prompt,
                    temperature=current_persona.temperature,
                    max_tokens=max_output_for_request
                )
                
                tokens_used = input_tokens + output_tokens
                cost_this_step = self.gemini_provider.calculate_usd_cost(input_tokens, output_tokens)
                
                self.intermediate_steps[current_output_key] = response_text
                # Store token usage for this specific step
                self.intermediate_steps[f"{current_output_key.replace('_Output', '').replace('_Critique', '').replace('_Feedback', '')}_Tokens_Used"] = tokens_used
                
                self.cumulative_token_usage += tokens_used
                self.cumulative_usd_cost += cost_this_step

                if kwargs.get('update_current_thought'): self.current_thought = response_text
                if kwargs.get('is_final_answer_step'): self.final_answer = response_text
                
                self._update_status(f"{current_persona_name} completed. Used {tokens_used} tokens.",
                                    current_total_tokens=self.cumulative_token_usage,
                                    current_total_cost=self.cumulative_usd_cost)
                return response_text
            except LLMProviderError as e:
                error_msg = f"[ERROR] Persona '{current_persona_name}' failed: {e}"
                self.intermediate_steps[current_output_key] = error_msg
                self._update_status(error_msg, state="error")
                if attempt == max_retries_on_fail: # If last attempt failed, re-raise
                    raise # Re-raise the exception to be caught by the main handler
                # Else, loop for retry

    def run_debate(self, max_turns: int = 5) -> Tuple[str, Dict[str, Any]]:
        """Executes the full Socratic debate loop."""
        if not self.personas:
            return {"error": "No personas loaded. Cannot run debate."}, {} # Return empty dict for intermediate steps on error
        
        self._update_status("Starting Socratic Arbitration Loop...",
                            current_total_tokens=self.cumulative_token_usage,
                            current_total_cost=self.cumulative_usd_cost)
        
        # Step 1: Visionary Generation
        context_string = self.prepare_context(self.codebase_context, self.intermediate_steps.get('debate_history', [])) # Pass context and history
        def visionary_prompt_gen():
            return (f"USER PROMPT: {self.initial_prompt}\n\n"
                    f"CODEBASE CONTEXT:\n{context_string}\n\n"
                    f"INSTRUCTIONS:\n"
                    "1. **Analyze the provided codebase context thoroughly.** Understand its structure, style, patterns, dependencies, and overall logic.\n"
                    "2. **Based on your analysis and the user prompt, propose an initial implementation strategy or code snippet.** Your proposal should be consistent with the existing codebase.\n"
                    "3. **Ensure your proposed code fits naturally into the existing architecture and follows its conventions.**")
        
        visionary_output = self._execute_persona_step("Visionary_Generator", visionary_prompt_gen, "Visionary_Generator_Output", update_current_thought=True)

        # Step 2: Skeptical Critique
        def skeptical_prompt_gen():
            return f"Critique the following proposal from a highly skeptical, risk-averse perspective. Identify potential failure points, architectural flaws, or critical vulnerabilities:\n\n{visionary_output}"
        skeptical_critique = self._execute_persona_step("Skeptical_Generator", skeptical_prompt_gen, "Skeptical_Critique")

        # Step 3: Domain-Specific Critiques
        domain_critiques_text = ""
        # Define core personas to exclude from domain-specific critiques
        core_personas = {"Visionary_Generator", "Skeptical_Generator", "Constructive_Critic", "Impartial_Arbitrator", "Devils_Advocate", "Generalist_Assistant"}
        # Get names of personas defined in the current domain's set, excluding core ones
        domain_expert_names = [p_name for p_name in self.personas if p_name not in core_personas]

        for expert_name in domain_expert_names:
            def expert_prompt_gen(name=expert_name, proposal=visionary_output):
                return (f"As a {name.replace('_', ' ')}, analyze the following proposal from your expert perspective. "
                        f"Identify specific points of concern, risks, or areas for improvement relevant to your domain. "
                        f"Your insights will be crucial for subsequent synthesis and refinement steps, so be thorough and specific. "
                        f"Present your analysis in a structured format, using clear headings or bullet points for 'Concerns' and 'Recommendations'.\n\n"
                        f"Proposal:\n{proposal}")
            
            critique = self._execute_persona_step(expert_name, expert_prompt_gen, f"{expert_name}_Critique")
            domain_critiques_text += f"\n\n--- {expert_name.replace('_', ' ')} Critique ---\n{critique}"

        # Step 4: Constructive Synthesis
        def constructive_prompt_gen():
            return (
                f"Original Proposal:\n{visionary_output}\n\n"
                f"--- Skeptical Critique ---\n{skeptical_critique}\n"
                f"{domain_critiques_text}\n\n"
                f"Based on all the above inputs, provide specific, actionable improvements. Synthesize the critiques, resolve conflicts where possible, and propose a refined solution or code.")
        constructive_feedback = self._execute_persona_step("Constructive_Critic", constructive_prompt_gen, "Constructive_Feedback")

        # Step 5: Final Arbitration
        def arbitrator_prompt_gen():
            # Revert the escaping instruction to the more robust version
            return (
                f"Synthesize all the following information into a single, balanced, and definitive final answer. Your output MUST be a JSON object with the following structure:\n\n```json\n{{\n  \"COMMIT_MESSAGE\": \"<string>\",\n  \"RATIONALE\": \"<string, including CONFLICT RESOLUTION: or UNRESOLVED CONFLICT: if applicable>\",\n  \"CODE_CHANGES\": [\n    {{\n      \"file_path\": \"<string>\",\n      \"action\": \"ADD | MODIFY | REMOVE\",\n      \"full_content\": \"<string>\" (Required for ADD/MODIFY actions, representing the entire new file content or modified file content. ENSURE ALL DOUBLE QUOTES WITHIN THE CONTENT ARE ESCAPED AS \\\".)\n    }},\n    {{\n      \"file_path\": \"<string>\",\n      \"action\": \"REMOVE\",\n      \"lines\": [\"<string>\", \"<string>\"] (Required for REMOVE action, representing the specific lines to be removed)\n    }}\n  ]\n}}\n```\n\nEnsure that the `CODE_CHANGES` array contains objects for each file change. For `MODIFY` and `ADD` actions, provide the `full_content` of the file. For `REMOVE` actions, provide an array of `lines` to be removed. If there are conflicting suggestions, you must identify them and explain your resolution in the 'RATIONALE' section, starting with 'CONFLICT RESOLUTION: '. If a conflict cannot be definitively resolved or requires further human input, flag it clearly in the 'RATIONALE' starting with 'UNRESOLVED CONFLICT: '."
                f"--- DEBATE SUMMARY ---\n"
                f"User Prompt: {self.initial_prompt}\n\n"
                f"Visionary Proposal:\n{visionary_output}\n\n"
                f"Skeptical Critique:\n{skeptical_critique}\n"
                f"{domain_critiques_text}\n\n"
                f"Constructive Feedback:\n{constructive_feedback}\n\n"
                f"--- END DEBATE ---"
            )
        self._execute_persona_step("Impartial_Arbitrator", arbitrator_prompt_gen, "Arbitrator_Output", is_final_answer_step=True)

        # Step 6: Devil's Advocate
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