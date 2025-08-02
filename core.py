# core.py
import yaml
import time
import sys # Added import for sys
import re
import ast
import pycodestyle
import difflib
import hashlib
import subprocess
import tempfile
import os
import json # Added import for JSON parsing
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
    CONTEXT_TOKEN_BUDGET_RATIO = 0.25 # Allocate 25% of total budget for context analysis

    def __init__(self,
                 initial_prompt: str, # Renamed from 'prompt' to 'initial_prompt'
                 api_key: str,
                 max_total_tokens_budget: int,
                 model_name: str,
                 personas: Dict[str, Persona],
                 all_personas: Dict[str, Persona],
                 persona_sets: Dict[str, List[str]],
                 domain: str = "General",
                 status_callback: Callable = None,
                 rich_console: Optional[Console] = None,
                 codebase_context: Optional[Dict[str, str]] = None):
        self.initial_prompt = initial_prompt
        self.max_total_tokens_budget = max_total_tokens_budget
        self.model_name = model_name
        self.personas = personas
        self.domain = domain
        self.all_personas = all_personas
        self.persona_sets = persona_sets
        self.status_callback = status_callback
        self.gemini_provider = GeminiProvider(api_key=api_key, model_name=model_name, status_callback=self._update_status)
        self.cumulative_token_usage = 0
        self.cumulative_usd_cost = 0.0
        self.intermediate_steps: Dict[str, Any] = {}
        self.rich_console = rich_console if rich_console else Console()
        self.current_thought = initial_prompt
        self.final_answer = "Process did not complete."
        self.codebase_context = codebase_context

    def _update_status(self, message: str, **kwargs):
        self.rich_console.print(message)
        if self.status_callback:
            self.status_callback(message=message, **kwargs)

    def _get_persona(self, name: str) -> Persona:
        persona = self.personas.get(name) or self.all_personas.get(name)
        if not persona:
            raise ValueError(f"Persona '{name}' not found.")
        return persona

    def _prioritize_python_code(self, content: str, max_tokens: int) -> str:
        """Prioritizes imports, class/function definitions for Python code."""
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
            self.rich_console.print(f"[yellow]Warning: Syntax error in Python context file, falling back to simple truncation.[/yellow]")
            return self._truncate_text_by_tokens(content, max_tokens)

        # Combine and truncate
        combined_content = "\n".join(priority_lines + other_lines)
        return self._truncate_text_by_tokens(combined_content, max_tokens)

    def _truncate_text_by_tokens(self, text: str, max_tokens: int) -> str:
        """Truncates text to fit within max_tokens."""
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
        
        # Refine by token count
        while self.gemini_provider.count_tokens(truncated_text, "") > max_tokens and len(truncated_text) > 0:
            truncated_text = truncated_text[:-max(1, len(truncated_text) // 20)] # Remove 5% or at least 1 char
            if len(truncated_text) == 0:
                break
        
        # Add an ellipsis if truncated
        if self.gemini_provider.count_tokens(text, "") > max_tokens:
            return truncated_text.strip() + "\n... (truncated)"
        return truncated_text

    def _prepare_context_for_prompt(self) -> str:
        if not self.codebase_context:
            return "No codebase context provided."

        context_budget = self.max_total_tokens_budget * self.CONTEXT_TOKEN_BUDGET_RATIO
        context_str_parts = []
        current_tokens = 0

        for path, content in self.codebase_context.items():
            header = f"--- file_path: {path} ---\n"
            header_tokens = self.gemini_provider.count_tokens(header, "")
            
            remaining_budget_for_file_content = context_budget - current_tokens - header_tokens
            if remaining_budget_for_file_content <= 0:
                self._update_status(f"Skipping file '{path}' due to context token budget.")
                break

            file_content_to_add = ""
            if path.endswith('.py'):
                file_content_to_add = self._prioritize_python_code(content, remaining_budget_for_file_content)
            else:
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

    def _execute_persona_step(self, persona_name: str, step_prompt_generator: Callable[[], str], output_key: str, **kwargs) -> str:
        persona = self._get_persona(persona_name)
        step_prompt = step_prompt_generator()
        
        estimated_input_tokens = self.gemini_provider.count_tokens(prompt=step_prompt, system_prompt=persona.system_prompt)
        remaining_budget = self.max_total_tokens_budget - self.cumulative_token_usage
        
        if estimated_input_tokens >= remaining_budget:
            raise TokenBudgetExceededError(f"Prompt for '{persona_name}' ({estimated_input_tokens} tokens) exceeds remaining budget ({remaining_budget} tokens).")

        max_output_for_request = min(persona.max_tokens, max(0, remaining_budget - estimated_input_tokens))
        
        self._update_status(f"Running persona: {persona.name}...",
                            current_total_tokens=self.cumulative_token_usage,
                            current_total_cost=self.cumulative_usd_cost,
                            estimated_next_step_tokens=estimated_input_tokens + max_output_for_request,
                            estimated_next_step_cost=self.gemini_provider.calculate_usd_cost(estimated_input_tokens, max_output_for_request))
        
        try:
            response_text, input_tokens, output_tokens = self.gemini_provider.generate(
                prompt=step_prompt,
                system_prompt=persona.system_prompt,
                temperature=persona.temperature,
                max_tokens=max_output_for_request
            )
            
            tokens_used = input_tokens + output_tokens
            cost_this_step = self.gemini_provider.calculate_usd_cost(input_tokens, output_tokens)
            
            self.intermediate_steps[output_key] = response_text
            self.intermediate_steps[f"{output_key.replace('_Output', '')}_Tokens_Used"] = tokens_used
            self.cumulative_token_usage += tokens_used
            self.cumulative_usd_cost += cost_this_step

            if kwargs.get('update_current_thought'): self.current_thought = response_text
            if kwargs.get('is_final_answer_step'): self.final_answer = response_text
            
            self._update_status(f"{persona.name} completed. Used {tokens_used} tokens.",
                                current_total_tokens=self.cumulative_token_usage,
                                current_total_cost=self.cumulative_usd_cost)
            return response_text
        except LLMProviderError as e:
            error_msg = f"[ERROR] Persona '{persona_name}' failed: {e}"
            self.intermediate_steps[output_key] = error_msg
            self._update_status(error_msg, state="error")
            raise

    def run_debate(self) -> Tuple[str, Dict[str, Any]]:
        self._update_status("Starting Socratic Arbitration Loop...",
                            current_total_tokens=self.cumulative_token_usage,
                            current_total_cost=self.cumulative_usd_cost)
        
        # Step 1: Visionary Generation
        context_string = self._prepare_context_for_prompt()
        def visionary_prompt_gen():
            return (f"USER PROMPT: {self.initial_prompt}\n\n"
                    f"CODEBASE CONTEXT:\n{context_string}\n\n"
                    "INSTRUCTIONS:\n"
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
        core_personas = {"Visionary_Generator", "Skeptical_Generator", "Constructive_Critic", "Impartial_Arbitrator", "Devils_Advocate", "Generalist_Assistant"}
        domain_experts = [p_name for p_name in self.personas if p_name not in core_personas]

        for expert_name in domain_experts:
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
            return (f"Original Proposal:\n{visionary_output}\n\n"
                    f"--- Skeptical Critique ---\n{skeptical_critique}\n"
                    f"{domain_critiques_text}\n\n"
                    "Based on all the above inputs, provide specific, actionable improvements. Synthesize the critiques, resolve conflicts where possible, and propose a refined solution or code.")
        constructive_feedback = self._execute_persona_step("Constructive_Critic", constructive_prompt_gen, "Constructive_Feedback")

        # Step 5: Final Arbitration
        def arbitrator_prompt_gen():
            return (
                f"Synthesize all the following information into a single, balanced, and definitive final answer. Your output MUST be a JSON object with the following structure:\n\n```json\n{{\n  \"COMMIT_MESSAGE\": \"<string>\",\n  \"RATIONALE\": \"<string, including CONFLICT RESOLUTION: or UNRESOLVED CONFLICT: if applicable>\",\n  \"CODE_CHANGES\": [\n    {{\n      \"file_path\": \"<string>\",\n      \"action\": \"ADD | MODIFY | REMOVE\",\n      \"full_content\": \"<string>\" (Required for ADD/MODIFY actions, representing the entire new file content or modified file content)\n    }},\n    {{\n      \"file_path\": \"<string>\",\n      \"action\": \"REMOVE\",\n      \"lines\": [\"<string>\", \"<string>\"] (Required for REMOVE action, representing the specific lines to be removed)\n    }}\n  ]\n}}\n```\n\nEnsure that the `CODE_CHANGES` array contains objects for each file change. For `MODIFY` and `ADD` actions, provide the `full_content` of the file. For `REMOVE` actions, provide an array of `lines` to be removed. If there are conflicting suggestions, you must identify them and explain your resolution in the 'RATIONALE' section, starting with 'CONFLICT RESOLUTION: '. If a conflict cannot be definitively resolved or requires further human input, flag it clearly in the 'RATIONALE' starting with 'UNRESOLVED CONFLICT: '.\n\n"
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

# --- Post-Processing and Validation Functions ---

def _run_validation_in_sandbox(command: List[str], content: str, timeout: int = 10) -> Tuple[int, str, str]: # Added str return type for temp_file_path
    """
    Executes a command in a sandboxed environment using a temporary file.
    Returns (return_code, stdout_stderr_output, temp_file_path).
    """
    temp_file_path = None # Initialize to None
    try:
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.py') as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name # Store the path
        
        # Replace a placeholder filename with the actual temp file path if present
        cmd_with_file = [arg.replace("TEMP_FILE_PLACEHOLDER", temp_file_path) for arg in command]
        
        # Ensure the Python executable is used explicitly
        if cmd_with_file[0] == "python":
            cmd_with_file[0] = sys.executable
        process = subprocess.run(
            cmd_with_file,
            capture_output=True,
            text=True,
            check=False, # Don't raise CalledProcessError for non-zero exit codes
            timeout=timeout,
            env={"PYTHONPATH": os.getcwd()} # Ensure Python can find local modules if needed
        )
        return process.returncode, process.stdout + process.stderr, temp_file_path # Return temp_file_path
    except subprocess.TimeoutExpired:
        return 1, f"Validation timed out after {timeout} seconds.", temp_file_path # Return temp_file_path
    except Exception as e:
        return 1, f"Error running validation command: {e}", temp_file_path # Return temp_file_path
    finally:
        if temp_file_path and os.path.exists(temp_file_path): # Check if path exists before removing
            os.remove(temp_file_path) # Clean up the temporary file

def parse_llm_code_output(llm_output: str) -> Dict[str, Any]:
    """Parses the structured JSON output from the LLM into a dictionary."""
    # Initialize output structure
    output = {
        'summary': {'commit_message': '', 'rationale': '', 'conflict_resolution': '', 'unresolved_conflict': ''},
        'changes': {},
        'malformed_blocks': []
    }
    
    # Pre-process: Remove markdown code block fences if present
    # This regex looks for ```json or ``` followed by content, and then ```
    # It's designed to extract the content *between* the fences.
    json_block_match = re.search(r'```json\s*(.*?)\s*```', llm_output, re.DOTALL)
    if json_block_match:
        llm_output_cleaned = json_block_match.group(1).strip()
    else:
        # If no ```json block found, try generic ``` block
        json_block_match = re.search(r'```\s*(.*?)\s*```', llm_output, re.DOTALL)
        if json_block_match:
            llm_output_cleaned = json_block_match.group(1).strip()
        else:
            llm_output_cleaned = llm_output.strip() # No fences, use as is

    try:
        json_data = json.loads(llm_output_cleaned) # Use the cleaned output here
        if not isinstance(json_data, dict):
            raise ValueError("LLM output is not a JSON object.")

        # Extract summary fields
        output['summary']['commit_message'] = json_data.get('COMMIT_MESSAGE', '').strip()
        rationale_content = json_data.get('RATIONALE', '').strip()
        output['summary']['rationale'] = rationale_content

        # Extract conflict resolution/unresolved conflict from rationale within JSON
        if rationale_content:
            conflict_res_match = re.search(r"CONFLICT RESOLUTION:\s*(.*?)(?=\nUNRESOLVED CONFLICT:|\n\n|$)", rationale_content, re.DOTALL)
            if conflict_res_match:
                output['summary']['conflict_resolution'] = conflict_res_match.group(1).strip()
            
            unresolved_conflict_match = re.search(r"UNRESOLVED CONFLICT:\s*(.*?)(?=\n\n|$)", rationale_content, re.DOTALL)
            if unresolved_conflict_match:
                output['summary']['unresolved_conflict'] = unresolved_conflict_match.group(1).strip()

        # Extract code changes
        code_changes_list = json_data.get('CODE_CHANGES', [])
        if not isinstance(code_changes_list, list):
            output['malformed_blocks'].append(f"CODE_CHANGES is not a list: {code_changes_list}")
            return output # Exit early if CODE_CHANGES is malformed

        for change_item in code_changes_list:
            if not isinstance(change_item, dict) or 'file_path' not in change_item or 'action' not in change_item:
                output['malformed_blocks'].append(f"Malformed change item: {change_item}")
                continue
            
            file_path = change_item['file_path']
            action = change_item['action']

            if action == 'ADD':
                if 'full_content' in change_item:
                    output['changes'][file_path] = {'type': 'ADD', 'content': change_item['full_content'].strip()}
                else:
                    output['malformed_blocks'].append(f"ADD action missing 'full_content' for {file_path}: {change_item}")
            elif action == 'MODIFY':
                if 'full_content' in change_item:
                    output['changes'][file_path] = {'type': 'MODIFY', 'new_content': change_item['full_content'].strip()}
                else:
                    output['malformed_blocks'].append(f"MODIFY action missing 'full_content' for {file_path}: {change_item}")
            elif action == 'REMOVE':
                if 'lines' in change_item and isinstance(change_item['lines'], list):
                    output['changes'][file_path] = {'type': 'REMOVE', 'lines': change_item['lines']}
                else:
                    output['malformed_blocks'].append(f"REMOVE action missing 'lines' or 'lines' not a list for {file_path}: {change_item}")
            else:
                output['malformed_blocks'].append(f"Unknown action type '{action}' for {file_path}: {change_item}")

    except json.JSONDecodeError as e:
        output['malformed_blocks'].append(f"LLM output is not valid JSON: {e}\nRaw output:\n{llm_output}")
    except ValueError as e:
        output['malformed_blocks'].append(f"JSON parsing error: {e}\nRaw output:\n{llm_output}")
    except Exception as e:
        output['malformed_blocks'].append(f"An unexpected error occurred during parsing: {e}\nRaw output:\n{llm_output}")
    
    return output

def validate_code_output(parsed_data: Dict, original_context: Dict) -> Dict:
    """Validates parsed code for syntax, style, and consistency using sandboxed execution."""
    report = {'issues': [], 'malformed_blocks': parsed_data.get('malformed_blocks', [])}

    for file_path, change in parsed_data.get('changes', {}).items():
        content_to_check = ""
        is_python = file_path.endswith('.py')

        if change['type'] == 'ADD':
            content_to_check = change['content']
            checksum = hashlib.sha256(content_to_check.encode('utf-8')).hexdigest()
            report['issues'].append({'type': 'Content Integrity', 'file': file_path, 'message': f"New file SHA256: {checksum}"})

        elif change['type'] == 'MODIFY':
            content_to_check = change['new_content']
            # Post-Diff Consistency: Calculate checksum of expected new content
            checksum_new = hashlib.sha256(content_to_check.encode('utf-8')).hexdigest()
            report['issues'].append({'type': 'Content Integrity', 'file': file_path, 'message': f"Modified file (new content) SHA256: {checksum_new}"})

        elif change['type'] == 'REMOVE':
            # No content to check for syntax/style for REMOVE, but can check if lines exist in original
            original_lines = original_context.get(file_path, "").splitlines()
            for line_to_remove in change['lines']:
                if line_to_remove not in original_lines:
                    report['issues'].append({'type': 'Diff Inconsistency', 'file': file_path, 'message': f"Line to remove '{line_to_remove}' not found in original file.", 'line': 'N/A'})
            continue # No syntax/style checks for REMOVE

        if is_python and content_to_check:
            # 1. Syntax Validation (sandboxed)
            # Use py_compile for robust syntax validation
            ast_command = [sys.executable, "-m", "py_compile", "TEMP_FILE_PLACEHOLDER"]
            # Capture the temp_file_path returned by the sandbox function
            ast_returncode, ast_output, _ = _run_validation_in_sandbox(ast_command, content_to_check)
            if ast_returncode != 0:
                report['issues'].append({'type': 'Syntax Error', 'file': file_path, 'message': ast_output.strip()})
            
            # 2. Style Compliance (PEP8) (sandboxed)
            # Use pycodestyle on a temporary file
            pep8_command = [sys.executable, "-m", "pycodestyle", "--format=default", "TEMP_FILE_PLACEHOLDER"]
            # Capture the temp_file_path returned by the sandbox function
            pep8_returncode, pep8_output, temp_file_for_pep8 = _run_validation_in_sandbox(pep8_command, content_to_check)
            if pep8_returncode != 0:
                # Parse pycodestyle output to get individual errors
                for line in pep8_output.splitlines():
                    # Use the actual temporary file path in the regex, escaping it for special characters
                    escaped_temp_path = re.escape(temp_file_for_pep8)
                    match = re.match(rf'{escaped_temp_path}:(\d+):\d+: (.+)', line)
                    if match:
                        report['issues'].append({'type': 'PEP8 Violation', 'file': file_path, 'message': match.group(2), 'line': int(match.group(1))})
                    else:
                        report['issues'].append({'type': 'PEP8 Violation', 'file': file_path, 'message': line})
                
    return report

def format_git_diff(original_content: str, new_content: str) -> str:
    """Creates a git-style unified diff from original and new content."""
    original_lines = original_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    
    diff = difflib.unified_diff(
        original_lines,
        new_lines,
        fromfile='a/original', tofile='b/modified',
        lineterm='' # Prevent adding extra newlines if already present
    )
    # Skip the '--- a/original' and '+++ b/modified' headers
    return "".join(list(diff)[2:])

def run_isal_process(
    prompt: str, # This is the argument passed to run_isal_process
    api_key: str,
    max_total_tokens_budget: int = 10000,
    model_name: str = "gemini-2.5-flash-lite",
    domain: str = "auto",
    streamlit_status_callback: Callable = None,
    all_personas: Optional[Dict[str, Persona]] = None, # Pass all personas to core
    persona_sets: Optional[Dict[str, List[str]]] = None, # Pass all persona sets to core
    personas_override: Optional[Dict[str, Persona]] = None,
    rich_console: Optional[Console] = None, # New argument for rich console
    codebase_context: Optional[Dict[str, str]] = None # New argument for codebase context
) -> 'SocraticDebate':
    """Initializes and returns the SocraticDebate instance."""
    if personas_override:
        personas = personas_override
        domain = "Custom"
    else:
        # Load personas and sets if not provided (for CLI or initial app load)
        if all_personas is None or persona_sets is None:
            all_personas, persona_sets, default_set = load_personas()
        
        # Determine domain to use
        if domain == "auto" and prompt.strip() and api_key.strip(): # Only auto-recommend if prompt and key are present
            from llm_provider import recommend_domain
            # Note: recommend_domain in llm_provider.py uses a different prompt than app.py's keyword matching
            # It makes an LLM call.
            llm_recommended_domain = recommend_domain(prompt, api_key)
            if llm_recommended_domain in persona_sets:
                domain = llm_recommended_domain
            else:
                domain = default_set
        elif domain not in persona_sets:
            domain = default_set
        
        # Get the personas for the selected domain
        personas = {name: all_personas[name] for name in persona_sets[domain]}

    # Prepare kwargs for SocraticDebate.__init__
    kwargs_for_debate = {
        'initial_prompt': prompt, # Correctly map 'prompt' to 'initial_prompt'
        'api_key': api_key,
        'max_total_tokens_budget': max_total_tokens_budget,
        'model_name': model_name,
        'personas': personas,
        'all_personas': all_personas,
        'persona_sets': persona_sets,
        'domain': domain,
        'status_callback': streamlit_status_callback,
        'rich_console': rich_console,
        'codebase_context': codebase_context
    }
    
    debate = SocraticDebate(**kwargs_for_debate)
    return debate