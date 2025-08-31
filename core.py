# core.py
import json
import logging
import re
import uuid
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Callable, Optional, Type, Union
from functools import lru_cache # Keep lru_cache if used in methods, otherwise remove
from rich.console import Console
from pydantic import ValidationError

# --- IMPORT MODIFICATIONS ---
from src.llm_provider import GeminiProvider
from src.context.context_analyzer import ContextRelevanceAnalyzer
from src.persona.routing import PersonaRouter
from src.utils.output_parser import LLMOutputParser
from src.models import PersonaConfig, ReasoningFrameworkConfig, LLMOutput, CodeChange, ContextAnalysisOutput, CritiqueOutput, GeneralOutput, ConflictReport, SelfImprovementAnalysisOutput, ConfigurationAnalysisOutput, SelfImprovementAnalysisOutputV1, DeploymentAnalysisOutput
from src.config.settings import ChimeraSettings
from src.exceptions import ChimeraError, LLMResponseValidationError, SchemaValidationError, TokenBudgetExceededError, LLMProviderError, CircuitBreakerError
# Removed SELF_ANALYSIS_KEYWORDS, is_self_analysis_prompt as they are handled by PromptAnalyzer in PersonaManager
from src.logging_config import setup_structured_logging
from src.utils.error_handler import handle_errors
from src.persona_manager import PersonaManager
# NEW IMPORTS FOR SELF-IMPROVEMENT
from src.self_improvement.metrics_collector import ImprovementMetricsCollector
from src.self_improvement.content_validator import ContentAlignmentValidator

# Configure logging for the core module itself
logger = logging.getLogger(__name__)

class SocraticDebate:
    PERSONA_OUTPUT_SCHEMAS = {
        "Impartial_Arbitrator": LLMOutput,
        "Context_Aware_Assistant": ContextAnalysisOutput,
        "Constructive_Critic": CritiqueOutput,
        "General_Synthesizer": GeneralOutput,
        "Devils_Advocate": ConflictReport,
        "Self_Improvement_Analyst": SelfImprovementAnalysisOutputV1,
        "Code_Architect": CritiqueOutput,
        "Security_Auditor": CritiqueOutput,
        "DevOps_Engineer": CritiqueOutput,
        "Test_Engineer": CritiqueOutput,
    }

    def __init__(self, initial_prompt: str, api_key: str,
                 codebase_context: Optional[Dict[str, str]] = None,
                 settings: Optional[ChimeraSettings] = None,
                 all_personas: Optional[Dict[str, PersonaConfig]] = None,
                 persona_sets: Optional[Dict[str, List[str]]] = None,
                 domain: Optional[str] = None,
                 model_name: str = "gemini-2.5-flash-lite",
                 status_callback: Optional[Callable] = None,
                 rich_console: Optional[Console] = None,
                 context_analyzer: Optional[ContextRelevanceAnalyzer] = None,
                 is_self_analysis: bool = False,
                 persona_manager: Optional[PersonaManager] = None
                 ):
        """
        Initialize a Socratic debate session.
        """
        setup_structured_logging(log_level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.settings = settings or ChimeraSettings()
        self.max_total_tokens_budget = self.settings.total_budget
        self.tokens_used = 0
        self.model_name = model_name
        self.status_callback = status_callback
        self.rich_console = rich_console or Console(stderr=True)
        self.is_self_analysis = is_self_analysis

        self.request_id = str(uuid.uuid4())[:8]
        self._log_extra = {"request_id": self.request_id or "N/A"}

        self.initial_prompt = initial_prompt
        self.codebase_context = codebase_context or {}
        self.domain = domain
        self.context_analyzer = context_analyzer

        try:
            self.llm_provider = GeminiProvider(api_key=api_key, model_name=self.model_name, rich_console=self.rich_console, request_id=self.request_id)
        except LLMProviderError as e:
            self._log_with_context("error", f"Failed to initialize LLM provider: {e.message}", exc_info=True)
            raise ChimeraError(f"LLM provider initialization failed: {e.message}", original_exception=e) from e
        except Exception as e:
            self._log_with_context("error", f"An unexpected error occurred during LLM provider initialization: {e}", exc_info=True)
            raise ChimeraError(f"LLM provider initialization failed unexpectedly: {e}", original_exception=e) from e

        try:
            self.tokenizer = self.llm_provider.tokenizer
        except AttributeError:
            raise ChimeraError("LLM provider tokenizer is not available.")

        self.persona_manager = persona_manager
        if not self.persona_manager:
            self.logger.warning("PersonaManager instance not provided to SocraticDebate. Initializing a new one. This might affect state persistence in UI.")
            self.persona_manager = PersonaManager({})
            self.all_personas = self.persona_manager.all_personas
            self.persona_sets = self.persona_manager.persona_sets
        else:
            self.all_personas = self.persona_manager.all_personas
            self.persona_sets = self.persona_manager.persona_sets

        self.persona_router = self.persona_manager.persona_router
        if not self.persona_router:
             self.logger.warning("PersonaRouter not found in PersonaManager. Initializing a new one. This might indicate an issue in PersonaManager setup.")
             self.persona_router = PersonaRouter(self.all_personas, self.persona_sets, self.persona_manager.prompt_analyzer)

        self.content_validator = ContentAlignmentValidator(
            original_prompt=self.initial_prompt,
            debate_domain=self.domain
        )

        if self.codebase_context and self.context_analyzer:
            if isinstance(self.codebase_context, dict):
                try:
                    if not self.context_analyzer.file_embeddings:
                        self.context_analyzer.compute_file_embeddings(self.codebase_context)
                except Exception as e:
                    self._log_with_context("error", f"Failed to compute embeddings for codebase context: {e}", exc_info=True)
                    if self.status_callback:
                        self.status_callback(message=f"[red]Error computing context embeddings: {e}[/red]")
            else:
                self.logger.warning("codebase_context was not a dictionary, skipping embedding computation.")

        self._calculate_token_budgets()

    def _log_with_context(self, level: str, message: str, **kwargs):
        """Helper to add request context to all logs from this instance using the class-specific logger."""
        exc_info = kwargs.pop('exc_info', None)
        original_exception = kwargs.pop('original_exception', None)
        log_data = {**self._log_extra, **kwargs}

        for k, v in log_data.items():
            try:
                json.dumps({k: v})
            except TypeError:
                log_data[k] = str(v)

        logger_method = getattr(self.logger, level)
        if exc_info is not None:
            logger_method(message, exc_info=exc_info, extra=log_data)
        elif original_exception is not None:
            log_data['original_exception_type'] = type(original_exception).__name__
            log_data['original_exception_message'] = str(original_exception)
            logger_method(message, extra=log_data)
        else:
            logger_method(message, extra=log_data)

    def _calculate_token_budgets(self):
        """Calculates token budgets for different phases based on context, model limits, and prompt type."""
        try:
            prompt_analysis = self.persona_manager._analyze_prompt_complexity(self.initial_prompt)
            complexity_score = prompt_analysis['complexity_score']

            if self.is_self_analysis:
                context_ratio = self.settings.self_analysis_context_ratio
                debate_ratio = self.settings.self_analysis_debate_ratio
                synthesis_ratio = self.settings.self_analysis_synthesis_ratio
            else:
                base_context_ratio = self.settings.context_token_budget_ratio
                base_output_ratio = self.settings.synthesis_token_budget_ratio

                if 'code' in self.initial_prompt.lower() or 'implement' in self.initial_prompt.lower():
                    context_ratio = max(0.4, base_context_ratio - 0.15)
                    synthesis_ratio = min(0.5, base_output_ratio + 0.15)
                elif complexity_score > 0.7:
                    context_ratio = min(0.7, base_context_ratio + 0.15)
                    synthesis_ratio = max(0.2, base_output_ratio - 0.05)
                else:
                    context_ratio = base_context_ratio
                    synthesis_ratio = base_output_ratio

                debate_ratio = 1.0 - context_ratio - synthesis_ratio
                if debate_ratio < 0.05:
                    debate_ratio = 0.05
                    remaining_for_context_synthesis = 1.0 - debate_ratio
                    context_ratio = context_ratio / (context_ratio + synthesis_ratio) * remaining_for_context_synthesis
                    synthesis_ratio = synthesis_ratio / (context_ratio + synthesis_ratio) * remaining_for_context_synthesis

            total_current_ratios = context_ratio + debate_ratio + synthesis_ratio
            if total_current_ratios > 0 and abs(total_current_ratios - 1.0) > 1e-6:
                self._log_with_context("warning", f"Token budget ratios sum to {total_current_ratios}, normalizing them.",
                                       original_context_ratio=context_ratio, original_debate_ratio=debate_ratio,
                                       original_synthesis_ratio=synthesis_ratio)
                normalization_factor = 1.0 / total_current_ratios
                context_ratio *= normalization_factor
                debate_ratio *= normalization_factor
                synthesis_ratio *= normalization_factor
                self._log_with_context("debug", f"Normalized ratios: context={context_ratio}, debate={debate_ratio}, synthesis={synthesis_ratio}")

            context_str = self.context_analyzer.get_context_summary() if self.context_analyzer else ""
            self.initial_input_tokens = self.tokenizer.count_tokens(context_str + self.initial_prompt)

            remaining_tokens = max(0, self.max_total_tokens_budget - self.initial_input_tokens)

            context_tokens_budget = int(remaining_tokens * context_ratio)
            debate_tokens_budget = int(remaining_tokens * debate_ratio)
            synthesis_tokens_budget = int(remaining_tokens * synthesis_ratio)

            MIN_PHASE_TOKENS = 250

            context_tokens_budget = max(MIN_PHASE_TOKENS, context_tokens_budget)
            debate_tokens_budget = max(MIN_PHASE_TOKENS, debate_tokens_budget)
            synthesis_tokens_budget = max(MIN_PHASE_TOKENS, synthesis_tokens_budget)

            total_allocated = context_tokens_budget + debate_tokens_budget + synthesis_tokens_budget
            if total_allocated > remaining_tokens:
                reduction_factor = remaining_tokens / total_allocated
                context_tokens_budget = int(context_tokens_budget * reduction_factor)
                debate_tokens_budget = int(debate_tokens_budget * reduction_factor)
                synthesis_tokens_budget = int(synthesis_tokens_budget * reduction_factor)

                context_tokens_budget = max(MIN_PHASE_TOKENS, context_tokens_budget)
                debate_tokens_budget = max(MIN_PHASE_TOKENS, debate_tokens_budget)
                synthesis_tokens_budget = max(MIN_PHASE_TOKENS, synthesis_tokens_budget)

            self.phase_budgets = {
                "context": context_tokens_budget,
                "debate": debate_tokens_budget,
                "synthesis": synthesis_tokens_budget,
                "persona_turn_budgets": {}
            }

            self._log_with_context("info", "SocraticDebate token budgets initialized",
                                   initial_input_tokens=self.initial_input_tokens,
                                   context_budget=self.phase_budgets["context"],
                                   debate_budget=self.phase_budgets["debate"],
                                   synthesis_budget=self.phase_budgets["synthesis"],
                                   max_total_tokens_budget=self.max_total_tokens_budget,
                                   prompt_complexity=prompt_analysis,
                                   is_self_analysis=self.is_self_analysis)

        except Exception as e:
            self._log_with_context("error", "Token budget calculation failed",
                                   error=str(e), context="token_budget", exc_info=True, original_exception=e)
            self.phase_budgets = {"context": 500, "debate": 15000, "synthesis": 1000, "persona_turn_budgets": {}}
            self.initial_input_tokens = 0
            raise ChimeraError("Failed to calculate token budgets due to an unexpected error.", original_exception=e) from e

    def track_token_usage(self, phase: str, tokens: int):
        """Tracks token usage for a given phase."""
        self.tokens_used += tokens
        cost = self.llm_provider.calculate_usd_cost(tokens, 0)
        self.intermediate_steps.setdefault(f"{phase}_Tokens_Used", 0)
        self.intermediate_steps[f"{phase}_Tokens_Used"] += tokens
        self.intermediate_steps.setdefault(f"{phase}_Estimated_Cost_USD", 0.0)
        self.intermediate_steps[f"{phase}_Estimated_Cost_USD"] += cost
        self._log_with_context("debug", f"Tokens used in {phase}: {tokens}. Total: {self.tokens_used}",
                               phase=phase, tokens_added=tokens, total_tokens=self.tokens_used)

    def check_budget(self, phase: str, tokens_needed: int, step_name: str):
        """Checks if adding tokens for the next step would exceed the budget."""
        if self.tokens_used + tokens_needed > self.max_total_tokens_budget:
            self._log_with_context("warning", f"Token budget exceeded for {step_name} in {phase} phase.",
                                   current_tokens=self.tokens_used, tokens_needed=tokens_needed,
                                   budget=self.max_total_tokens_budget, step=step_name, phase=phase)
            raise TokenBudgetExceededError(self.tokens_used, self.max_total_tokens_budget,
                                           details={"phase": phase, "step_name": step_name, "tokens_needed": tokens_needed})

    def get_total_used_tokens(self) -> int:
        """Returns the total tokens used so far."""
        return self.tokens_used

    def get_total_estimated_cost(self) -> float:
        """Returns the total estimated cost so far."""
        total_cost = 0.0
        for key, value in self.intermediate_steps.items():
            if key.endswith("_Estimated_Cost_USD"):
                total_cost += value
        return total_cost

    def get_progress_pct(self, phase: str, completed: bool = False, error: bool = False) -> float:
        """Calculates the progress percentage for the debate."""
        phase_weights = {
            "context": 0.1,
            "debate": 0.7,
            "synthesis": 0.2
        }

        current_progress = 0.0
        if phase == "context":
            current_progress = 0.05
        elif phase == "debate":
            total_debate_personas = len(self.intermediate_steps.get("Persona_Sequence", [])) - 1
            completed_debate_personas = sum(1 for k in self.intermediate_steps if k.endswith("_Output") and not k.startswith(("Final_", "Context_Analysis_Output")) and k != "Self_Improvement_Metrics" and k != "Debate_History" and k != "Conflict_Resolution_Attempt" and k != "Unresolved_Conflict")

            if total_debate_personas > 0:
                current_progress = phase_weights["context"] + (completed_debate_personas / total_debate_personas) * phase_weights["debate"]
            else:
                current_progress = phase_weights["context"]
        elif phase == "synthesis":
            current_progress = phase_weights["context"] + phase_weights["debate"] + 0.1

        if completed:
            current_progress = 1.0
        elif error:
            current_progress = max(current_progress, 0.99)

        return min(max(0.0, current_progress), 1.0)

    def _execute_llm_turn(self, persona_name: str, persona_config: PersonaConfig,
                          prompt_for_llm: str, phase: str,
                          max_output_tokens_for_turn: int,
                          requested_model_name: Optional[str] = None,
                          max_retries: int = 2) -> Dict[str, Any]:
        """
        Executes a single LLM turn for a given persona, handles API calls,
        parsing, validation, and token tracking, with retry logic for validation failures.
        """
        self._log_with_context("debug", f"Executing LLM turn for {persona_name} in {phase} phase.",
                               persona=persona_name, phase=phase)

        output_schema = self.PERSONA_OUTPUT_SCHEMAS.get(persona_name, GeneralOutput)
        self._log_with_context("debug", f"Using schema {output_schema.__name__} for {persona_name}.")

        raw_llm_output = ""
        input_tokens = 0
        output_tokens = 0
        has_schema_error = False
        is_truncated = False
        has_content_misalignment = False

        current_prompt = prompt_for_llm

        for attempt in range(max_retries + 1):
            try:
                if self.status_callback:
                    self.status_callback(message=f"LLM Call: [bold]{persona_name.replace('_', ' ')}[/bold] generating response (Attempt {attempt + 1}/{max_retries + 1})...",
                                         state="running",
                                         current_total_tokens=self.tokens_used,
                                         current_total_cost=self.get_total_estimated_cost(),
                                         progress_pct=self.get_progress_pct(phase),
                                         current_persona_name=persona_name)

                actual_model_max_output_tokens = self.llm_provider.tokenizer.max_output_tokens
                effective_max_output_tokens = min(max_output_tokens_for_turn, actual_model_max_output_tokens)
                self._log_with_context("debug", f"Adjusting max_output_tokens for {persona_name}. Requested: {max_output_tokens_for_turn}, Model Max: {actual_model_max_output_tokens}, Effective: {effective_max_output_tokens}")

                raw_llm_output, input_tokens, output_tokens = self.llm_provider.generate(
                    prompt=current_prompt,
                    system_prompt=persona_config.system_prompt,
                    temperature=persona_config.temperature,
                    max_tokens=effective_max_output_tokens,
                    persona_config=persona_config,
                    requested_model_name=requested_model_name or self.model_name
                )

                self.track_token_usage(phase, input_tokens + output_tokens)

                self.intermediate_steps[f"{persona_name}_Actual_Temperature"] = persona_config.temperature
                self.intermediate_steps[f"{persona_name}_Actual_Max_Tokens"] = effective_max_output_tokens

                if output_tokens >= effective_max_output_tokens * 0.95:
                    is_truncated = True
                    self._log_with_context("warning", f"Output for {persona_name} might be truncated. Output tokens ({output_tokens}) close to max_tokens ({effective_max_output_tokens}).")

                parser = LLMOutputParser()
                parsed_output = parser.parse_and_validate(raw_llm_output, output_schema)

                if parsed_output.get("malformed_blocks"):
                    has_schema_error = True
                    self.intermediate_steps.setdefault("malformed_blocks", []).extend(parsed_output["malformed_blocks"])
                    self._log_with_context("warning", f"Parser reported malformed blocks for {persona_name}.",
                                           persona=persona_name, malformed_blocks=parsed_output["malformed_blocks"])

                break

            except (LLMProviderError, TokenBudgetExceededError, CircuitBreakerError, ChimeraError) as e:
                self._log_with_context("error", f"Non-retryable error during LLM turn for {persona_name}: {e}",
                                       persona=persona_name, phase=phase, exc_info=True, original_exception=e)
                if self.persona_manager:
                    self.persona_manager.record_persona_performance(persona_name, attempt + 1, raw_llm_output, False, f"Non-retryable error: {type(e).__name__}")
                raise e

            except SchemaValidationError as e:
                if attempt < max_retries:
                    self._log_with_context("warning", f"Validation error for {persona_name} (Attempt {attempt + 1}/{max_retries + 1}). Retrying. Error: {e}",
                                           persona=persona_name, phase=phase, exc_info=True, original_exception=e)
                    
                    error_details = e.details if hasattr(e, 'details') and isinstance(e.details, dict) else {}
                    error_type = error_details.get('error_type', 'Unknown validation error')
                    field_path = error_details.get('field_path', 'N/A')
                    invalid_value_snippet = str(error_details.get('invalid_value', 'N/A'))[:200]

                    retry_feedback = f"PREVIOUS OUTPUT INVALID: {error_type} at '{field_path}'. Problematic value snippet: '{invalid_value_snippet}'.\n"
                    retry_feedback += "CRITICAL: Your output failed schema validation. You MUST correct this.\n"
                    retry_feedback += "REMEMBER: OUTPUT MUST BE RAW JSON ONLY WITH NO MARKDOWN. STRICTLY ADHERE TO THE SCHEMA.\n\n"
                    
                    current_prompt = f"{retry_feedback}Original prompt: {prompt_for_llm}"
          
                    self.intermediate_steps.setdefault("malformed_blocks", []).append({
                        "type": "RETRYABLE_VALIDATION_ERROR",
                        "message": str(e),
                        "attempt": attempt + 1,
                        "persona": persona_name
                    })
                    
                    continue
                else:
                    self._log_with_context("error", f"Max retries ({max_retries}) reached for {persona_name}. Returning fallback JSON.", persona=persona_name)
                    if self.persona_manager:
                        self.persona_manager.record_persona_performance(persona_name, attempt + 1, raw_llm_output, False, "Schema validation failed after multiple attempts")
                    return {
                        "ANALYSIS_SUMMARY": "JSON validation failed after multiple attempts",
                        "IMPACTFUL_SUGGESTIONS": [{
                            "AREA": "Robustness",
                            "PROBLEM": f"Failed to produce valid JSON after {max_retries} attempts",
                            "PROPOSED_SOLUTION": "Review system prompts and validation logic",
                            "IMPACT": "Critical failure in self-improvement process"
                        }]
                    }
            
            except Exception as e:
                self._log_with_context("error", f"An unexpected error occurred during LLM turn for {persona_name}: {e}",
                                       persona=persona_name, phase=phase, exc_info=True, original_exception=e)
                if self.persona_manager:
                    self.persona_manager.record_persona_performance(persona_name, attempt + 1, raw_llm_output, False, f"Unexpected error: {type(e).__name__}")
                raise ChimeraError(f"Unexpected error in LLM turn for {persona_name}: {e}", original_exception=e) from e

        self.intermediate_steps[f"{persona_name}_Output"] = parsed_output
        self.intermediate_steps[f"{persona_name}_Tokens_Used"] = input_tokens + output_tokens
        self.intermediate_steps[f"{persona_name}_Estimated_Cost_USD"] = self.llm_provider.calculate_usd_cost(input_tokens, output_tokens)

        self._log_with_context("info", f"LLM turn for {persona_name} completed successfully.",
                               persona=persona_name, phase=phase,
                               input_tokens=input_tokens, output_tokens=output_tokens,
                               total_tokens_for_turn=input_tokens + output_tokens)
        
        if self.persona_manager:
            is_aligned, validation_message, nuanced_feedback = self.content_validator.validate(persona_name, parsed_output)
            self.persona_manager.record_persona_performance(persona_name, attempt + 1, parsed_output, is_aligned and not has_schema_error, validation_message)
            
            if not is_aligned:
                has_content_misalignment = True
                self._log_with_context("warning", f"Content misalignment detected for {persona_name}: {validation_message}",
                                       persona=persona_name, validation_message=validation_message)
                self.intermediate_steps.setdefault("malformed_blocks", []).extend([{
                    "type": "CONTENT_MISALIGNMENT",
                    "message": f"Output from {persona_name} drifted from the core topic: {validation_message}",
                    "persona": persona_name,
                    "raw_string_snippet": str(parsed_output)[:500]
                }] + nuanced_feedback.get("malformed_blocks", []))
                
                if isinstance(parsed_output, dict):
                    parsed_output["content_misalignment_warning"] = validation_message
                else:
                    parsed_output = f"WARNING: Content misalignment detected: {validation_message}\n\n{parsed_output}"

        return parsed_output

    def _initialize_debate_state(self):
        """Initializes or resets the debate's internal state variables."""
        self.intermediate_steps = {}
        self.tokens_used = 0
        self.rich_console.print(f"[bold green]Starting Socratic Debate for prompt:[/bold green] [italic]{self.initial_prompt}[/italic]")
        self._log_with_context("info", "Debate state initialized.")

    def _perform_context_analysis(self, persona_sequence: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
        """
        Performs context analysis based on the initial prompt and codebase context.
        """
        if not self.codebase_context or not self.context_analyzer:
            self._log_with_context("info", "No codebase context or analyzer available. Skipping context analysis.")
            return None

        self._log_with_context("info", "Performing context analysis.")
        try:
            relevant_files = self.context_analyzer.find_relevant_files(
                self.initial_prompt,
                max_context_tokens=self.phase_budgets["context"],
                active_personas=persona_sequence
            )

            context_summary_str = self.context_analyzer.generate_context_summary(
                [f[0] for f in relevant_files],
                self.phase_budgets["context"],
                self.initial_prompt
            )

            estimated_context_tokens = self.tokenizer.count_tokens(context_summary_str)

            self.check_budget("context", estimated_context_tokens, "Context Analysis Summary")
            self.track_token_usage("context", estimated_context_tokens)

            context_analysis_output = {
                "relevant_files": relevant_files,
                "context_summary": context_summary_str,
                "estimated_tokens": estimated_context_tokens
            }
            self._log_with_context("info", "Context analysis completed.",
                                   relevant_files=[f[0] for f in relevant_files],
                                   estimated_tokens=estimated_context_tokens)
            return context_analysis_output
        except Exception as e:
            self._log_with_context("error", f"Error during context analysis: {e}", exc_info=True, original_exception=e)
            self.rich_console.print(f"[red]Error during context analysis: {e}[/red]")
            return {"error": f"Context analysis failed: {e}"}

    def _get_final_persona_sequence(self, prompt: str, context_analysis_results: Optional[Dict[str, Any]]) -> List[str]:
        """
        Delegates to the PersonaRouter to determine the optimal persona sequence,
        incorporating prompt analysis, domain, and context analysis results.
        """
        if not self.persona_router:
            self._log_with_context("error", "PersonaRouter not initialized. Falling back to default sequence.")
            return ["Visionary_Generator", "Skeptical_Generator", "Impartial_Arbitrator"]

        sequence = self.persona_router.determine_persona_sequence(
            prompt=prompt,
            domain=self.domain,
            intermediate_results=self.intermediate_steps,
            context_analysis_results=context_analysis_results
        )
        return sequence

    def _distribute_debate_persona_budgets(self, persona_sequence: List[str]):
        """
        Distributes the total debate token budget among the actual personas in the sequence.
        This is called *after* the final persona sequence is determined.
        """
        MIN_PERSONA_TOKENS = 256

        active_debate_personas = [
            p for p in persona_sequence
            if p not in ["Context_Aware_Assistant", "Impartial_Arbitrator", "General_Synthesizer", "Self_Improvement_Analyst"]
        ]

        num_debate_personas = len(active_debate_personas)
        if num_debate_personas == 0:
            self.phase_budgets["persona_turn_budgets"] = {}
            self.phase_budgets["debate"] = 0
            return

        base_allocation_per_persona = self.phase_budgets["debate"] // num_debate_personas
        remaining_budget = self.phase_budgets["debate"]

        persona_turn_budgets: Dict[str, int] = {}

        for p_name in active_debate_personas:
            persona_config = self.persona_manager.get_adjusted_persona_config(p_name)
            allocated = max(MIN_PERSONA_TOKENS, min(base_allocation_per_persona, persona_config.max_tokens))
            persona_turn_budgets[p_name] = allocated
            remaining_budget -= allocated

        if remaining_budget > 0:
            redistribution_pool_personas = [
                p_name for p_name in active_debate_personas
                if persona_turn_budgets[p_name] < self.persona_manager.get_adjusted_persona_config(p_name).max_tokens
            ]
            if redistribution_pool_personas:
                share_per_redistribution_persona = remaining_budget // len(redistribution_pool_personas)
                for p_name in redistribution_pool_personas:
                    persona_config = self.persona_manager.get_adjusted_persona_config(p_name)
                    persona_turn_budgets[p_name] = min(persona_config.max_tokens, persona_turn_budgets[p_name] + share_per_redistribution_persona)

        current_total_persona_budget = sum(persona_turn_budgets.values())
        if current_total_persona_budget > self.phase_budgets["debate"]:
            reduction_factor = self.phase_budgets["debate"] / current_total_persona_budget
            for p_name in active_debate_personas:
                persona_turn_budgets[p_name] = max(MIN_PERSONA_TOKENS, int(persona_turn_budgets[p_name] * reduction_factor))

        self.phase_budgets["persona_turn_budgets"] = persona_turn_budgets
        self._log_with_context("info", "Debate persona turn budgets distributed.",
                               persona_turn_budgets=self.phase_budgets["persona_turn_budgets"])


    def _process_context_persona_turn(self, persona_sequence: List[str], context_analysis_results: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Executes the Context_Aware_Assistant persona turn if it's in the sequence.
        """
        if "Context_Aware_Assistant" not in persona_sequence:
            self._log_with_context("info", "Context_Aware_Assistant not in sequence. Skipping turn.")
            return None

        self._log_with_context("info", "Executing Context_Aware_Assistant turn.")
        persona_config = self.all_personas.get("Context_Aware_Assistant")
        if not persona_config:
            self._log_with_context("error", "Context_Aware_Assistant persona configuration not found.")
            return {"error": "Context_Aware_Assistant config missing."}

        context_prompt_content = ""
        if context_analysis_results and context_analysis_results.get("relevant_files"):
            for file_path, _ in context_analysis_results["relevant_files"]:
                content = self.codebase_context.get(file_path, "")
                if content:
                    context_prompt_content += f"### File: {file_path}\n```\n{content}\n```\n\n"

        prompt = f"Analyze the following codebase context and provide a structured analysis:\n\n{context_prompt_content}"

        max_output_tokens_for_turn = self.phase_budgets.get("persona_turn_budgets", {}).get("Context_Aware_Assistant", persona_config.max_tokens)
        estimated_tokens = self.tokenizer.count_tokens(prompt) + max_output_tokens_for_turn
        self.check_budget("debate", estimated_tokens, "Context_Aware_Assistant")

        try:
            output = self._execute_llm_turn("Context_Aware_Assistant", persona_config, prompt, "debate", max_output_tokens_for_turn)
            self._log_with_context("info", "Context_Aware_Assistant turn completed.")
            return output
        except Exception as e:
            self._log_with_context("error", f"Error during Context_Aware_Assistant turn: {e}", exc_info=True, original_exception=e)
            self.rich_console.print(f"[red]Error during Context_Aware_Assistant turn: {e}[/red]")
            return {"error": f"Context_Aware_Assistant turn failed: {e}"}

    def _execute_debate_persona_turns(self, persona_sequence: List[str], context_persona_turn_results: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Executes the main debate turns for each persona in the sequence.
        """
        debate_history = []
        
        previous_output_for_llm: Union[str, Dict[str, Any]]
        if context_persona_turn_results:
            previous_output_for_llm = f"Initial Prompt: {self.initial_prompt}\n\nStructured Context Analysis:\n{json.dumps(context_persona_turn_results, indent=2)}"
        else:
            previous_output_for_llm = f"Initial Prompt: {self.initial_prompt}"


        personas_for_debate = [
            p for p in persona_sequence
            if p not in ["Context_Aware_Assistant", "Impartial_Arbitrator", "General_Synthesizer", "Self_Improvement_Analyst"]
        ]

        if "Devils_Advocate" in persona_sequence and "Devils_Advocate" not in personas_for_debate:
            insert_idx = len(personas_for_debate)
            for i, p_name in reversed(list(enumerate(personas_for_debate))):
                if "Critic" in p_name or "Analyst" in p_name or "Engineer" in p_name or "Architect" in p_name:
                    insert_idx = i + 1
                    break
            personas_for_debate.insert(insert_idx, "Devils_Advocate")

        total_debate_steps = len(personas_for_debate)
        for i, persona_name in enumerate(personas_for_debate):
            self._log_with_context("info", f"Executing debate turn for persona: {persona_name}", persona=persona_name)
            self.status_callback(
                message=f"Executing: [bold]{persona_name.replace('_', ' ')}[/bold]...",
                state="running",
                current_total_tokens=self.tokens_used,
                current_total_cost=self.get_total_estimated_cost(),
                progress_pct=self.get_progress_pct("debate"),
                current_persona_name=persona_name
            )

            persona_config = self.persona_manager.get_adjusted_persona_config(persona_name)
            if not persona_config:
                persona_config = self.all_personas.get(persona_name)
                if not persona_config:
                    self._log_with_context("error", f"Persona configuration not found for {persona_name}. Skipping turn.", persona=persona_name)
                    debate_history.append({"persona": persona_name, "error": "Config not found"})
                    continue

            persona_specific_context_str = ""
            if context_persona_turn_results:
                if persona_name == "Security_Auditor" and context_persona_turn_results.get("security_summary"):
                    persona_specific_context_str = f"Security Context Summary:\n{json.dumps(context_persona_turn_results['security_summary'], indent=2)}"
                elif persona_name == "Code_Architect" and context_persona_turn_results.get("architecture_summary"):
                    persona_specific_context_str = f"Architecture Context Summary:\n{json.dumps(context_persona_turn_results['architecture_summary'], indent=2)}"
                elif persona_name == "DevOps_Engineer" and context_persona_turn_results.get("devops_summary"):
                    persona_specific_context_str = f"DevOps Context Summary:\n{json.dumps(context_persona_turn_results['devops_summary'], indent=2)}"
                elif persona_name == "Test_Engineer" and context_persona_turn_results.get("testing_summary"):
                    persona_specific_context_str = f"Testing Context Summary:\n{json.dumps(context_persona_turn_results['testing_summary'], indent=2)}"
                elif context_persona_turn_results.get("general_overview"):
                    persona_specific_context_str = f"General Codebase Overview:\n{context_persona_turn_results['general_overview']}"
                if context_persona_turn_results.get("configuration_summary"):
                    persona_specific_context_str += f"\n\nStructured Configuration Analysis:\n{json.dumps(context_persona_turn_results['configuration_summary'], indent=2)}"
                if context_persona_turn_results.get("deployment_summary"):
                    persona_specific_context_str += f"\n\nStructured Deployment Robustness Analysis:\n{json.dumps(context_persona_turn_results['deployment_summary'], indent=2)}"


            current_prompt_parts = [f"Initial Problem: {self.initial_prompt}\n\n"]
            if persona_specific_context_str:
                current_prompt_parts.append(f"Relevant Code Context:\n{persona_specific_context_str}\n\n")

            is_previous_output_problematic = False
            if isinstance(previous_output_for_llm, dict):
                if previous_output_for_llm.get("malformed_blocks") or previous_output_for_llm.get("content_misalignment_warning"):
                    is_previous_output_problematic = True
            elif isinstance(previous_output_for_llm, str) and ("malformed" in previous_output_for_llm.lower() or "error" in previous_output_for_llm.lower()):
                is_previous_output_problematic = True

            if is_previous_output_problematic:
                summary_of_previous = "Previous persona's output had issues (malformed JSON or content misalignment). "
                if isinstance(previous_output_for_llm, dict):
                    if previous_output_for_llm.get("CRITIQUE_SUMMARY"):
                        summary_of_previous += f"Summary of previous critique: {previous_output_for_llm['CRITIQUE_SUMMARY']}"
                    elif previous_output_for_llm.get("ANALYSIS_SUMMARY"):
                        summary_of_previous += f"Summary of previous analysis: {previous_output_for_llm['ANALYSIS_SUMMARY']}"
                    elif previous_output_for_llm.get("general_output"):
                        summary_of_previous += f"Summary of previous general output: {previous_output_for_llm['general_output'][:200]}..."
                    else:
                        summary_of_previous += "Details in malformed_blocks."
                else:
                    summary_of_previous += f"Raw error snippet: {str(previous_output_for_llm)[:200]}..."
                current_prompt_parts.append(f"Previous Debate Output Summary (with issues):\n{summary_of_previous}\n\n")
            else:
                if isinstance(previous_output_for_llm, dict):
                    if previous_output_for_llm.get("CRITIQUE_SUMMARY"):
                        current_prompt_parts.append(f"Previous Debate Output:\n{json.dumps({'CRITIQUE_SUMMARY': previous_output_for_llm['CRITIQUE_SUMMARY'], 'SUGGESTIONS': previous_output_for_llm.get('SUGGESTIONS', [])}, indent=2)}\n\n")
                    elif previous_output_for_llm.get("ANALYSIS_SUMMARY"):
                        current_prompt_parts.append(f"Previous Debate Output:\n{json.dumps({'ANALYSIS_SUMMARY': previous_output_for_llm['ANALYSIS_SUMMARY'], 'IMPACTFUL_SUGGESTIONS': previous_output_for_llm.get('IMPACTFUL_SUGGESTIONS', [])}, indent=2)}\n\n")
                    elif previous_output_for_llm.get("general_output"):
                        current_prompt_parts.append(f"Previous Debate Output:\n{json.dumps({'general_output': previous_output_for_llm['general_output']}, indent=2)}\n\n")
                    else:
                        current_prompt_parts.append(f"Previous Debate Output:\n{json.dumps(previous_output_for_llm, indent=2)}\n\n")
                else:
                    current_prompt_parts.append(f"Previous Debate Output:\n{previous_output_for_llm}\n\n")

            current_prompt = "".join(current_prompt_parts)

            max_output_tokens_for_turn = self.phase_budgets.get("persona_turn_budgets", {}).get(persona_name, persona_config.max_tokens)
            estimated_tokens = self.tokenizer.count_tokens(current_prompt) + max_output_tokens_for_turn
            self.check_budget("debate", estimated_tokens, persona_name)

            try:
                output = self._execute_llm_turn(persona_name, persona_config, current_prompt, "debate", max_output_tokens_for_turn)

                if persona_name == "Devils_Advocate" and isinstance(output, dict):
                    try:
                        conflict_report = ConflictReport.model_validate(output)
                        if conflict_report.conflict_found:
                            resolution_result = self._trigger_conflict_sub_debate(conflict_report, debate_history)
                            if resolution_result and resolution_result.get("conflict_resolved"):
                                previous_output_for_llm = {"status": "conflict_resolved", "resolution": resolution_result["resolution_summary"]}
                                self.intermediate_steps["Conflict_Resolution_Attempt"] = resolution_result
                                self.intermediate_steps["Unresolved_Conflict"] = None
                            else:
                                self.intermediate_steps["Unresolved_Conflict"] = conflict_report.model_dump()
                                previous_output_for_llm = {"status": "conflict_unresolved", "conflict_report": conflict_report.model_dump()}
                        else:
                            self._log_with_context("info", f"Devils_Advocate reported no conflict: {conflict_report.summary}")
                            self.intermediate_steps["Unresolved_Conflict"] = None
                            self.intermediate_steps["Conflict_Resolution_Attempt"] = None
                            previous_output_for_llm = {"status": "no_conflict_reported", "summary": conflict_report.summary}
                    except ValidationError:
                        self._log_with_context("warning", f"Devils_Advocate output was not a valid ConflictReport. Treating as general output.")
                        previous_output_for_llm = output
                    except Exception as e:
                        self._log_with_context("error", f"Error processing Devils_Advocate output: {e}", exc_info=True)
                        previous_output_for_llm = {"error": f"Error processing Devils_Advocate output: {e}"}
                debate_history.append({"persona": persona_name, "output": output})
                previous_output_for_llm = output
            except Exception as e:
                self._log_with_context("error", f"Error during {persona_name} turn: {e}", persona=persona_name, exc_info=True, original_exception=e)
                self.rich_console.print(f"[red]Error during {persona_name} turn: {e}[/red]")
                previous_output_for_llm = {"error": f"Turn failed for {persona_name}: {str(e)}", "malformed_blocks": [{"type": "DEBATE_TURN_ERROR", "message": str(e)}]}
                continue

        self._log_with_context("info", "All debate turns completed.")
        return debate_history

    def _trigger_conflict_sub_debate(self, conflict_report: ConflictReport, debate_history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Triggers a focused sub-debate to resolve a specific conflict.
        Returns a resolution summary or None if unresolved.
        """
        self._log_with_context("info", f"Initiating sub-debate for conflict: {conflict_report.summary}")

        sub_debate_personas_set = set(conflict_report.involved_personas)

        if conflict_report.conflict_type in ["LOGICAL_INCONSISTENCY", "DATA_DISCREPANCY", "METHODOLOGY_DISAGREEMENT"]:
            if "Constructive_Critic" in self.all_personas:
                sub_debate_personas_set.add("Constructive_Critic")

        main_synthesis_persona = "Self_Improvement_Analyst" if self.is_self_analysis else \
                                 ("Impartial_Arbitrator" if self.domain == "Software Engineering" else "General_Synthesizer")
        if main_synthesis_persona in self.all_personas and main_synthesis_persona not in sub_debate_personas_set:
            sub_debate_personas_set.add(main_synthesis_persona)

        sub_debate_personas = [p for p in list(sub_debate_personas_set) if p in self.all_personas]

        if not sub_debate_personas:
            self._log_with_context("warning", "No relevant personas found for conflict sub-debate. Skipping resolution.")
            return None

        remaining_tokens = self.max_total_tokens_budget - self.tokens_used
        sub_debate_budget = max(1000, min(5000, int(remaining_tokens * 0.1)))

        if sub_debate_budget < 1000:
            self._log_with_context("warning", "Insufficient tokens for sub-debate, skipping conflict resolution.")
            return None

        sub_debate_prompt = f"""
        CRITICAL CONFLICT RESOLUTION REQUIRED:

        The following conflict has been identified in the main debate:
        Conflict Type: {conflict_report.conflict_type}
        Summary: {conflict_report.summary}
        Involved Personas: {', '.join(conflict_report.involved_personas)}
        Conflicting Outputs Snippet: {conflict_report.conflicting_outputs_snippet}
        Proposed Resolution Paths: {'; '.join(conflict_report.proposed_resolution_paths) if conflict_report.proposed_resolution_paths else 'None provided.'}

        Your task is to:
        1.  Analyze the conflicting perspectives from the debate history.
        2.  Identify the root cause of the disagreement.
        3.  Propose a definitive resolution or a clear compromise.
        4.  Provide a concise rationale for your resolution.

        Focus ONLY on resolving this specific conflict. Output a clear resolution summary.
        """

        resolution_attempts = []
        for persona_name in sub_debate_personas:
            self.status_callback(
                message=f"Sub-Debate: [bold]{persona_name.replace('_', ' ')}[/bold] resolving conflict...",
                state="running",
                current_total_tokens=self.tokens_used,
                current_total_cost=self.get_total_estimated_cost(),
                progress_pct=self.get_progress_pct("debate"),
                current_persona_name=persona_name
            )

            persona_config = self.persona_manager.get_adjusted_persona_config(persona_name)

            max_output_tokens_for_sub_debate_turn = sub_debate_budget // len(sub_debate_personas)
            max_output_tokens_for_sub_debate_turn = max(250, min(max_output_tokens_for_sub_debate_turn, persona_config.max_tokens))
            
            try:
                turn_output = self._execute_llm_turn(
                    persona_name, persona_config, sub_debate_prompt, "sub_debate_phase", max_output_tokens_for_sub_debate_turn
                )
                resolution_attempts.append({"persona": persona_name, "output": turn_output})
            except Exception as e:
                self._log_with_context("error", f"Sub-debate turn for {persona_name} failed: {e}", exc_info=True)
                self.rich_console.print(f"[red]Error during sub-debate turn for {persona_name}: {e}[/red]")
            finally:
                pass

        if resolution_attempts:
            final_resolution_prompt_base = f"""
            Synthesize the following sub-debate attempts to resolve a conflict into a single, clear resolution.
            Conflict: {conflict_report.summary}
            Sub-debate results: {json.dumps(resolution_attempts, indent=2)}

            Provide a final resolution and its rationale.
            """
            sub_debate_synthesis_budget = max(500, int(sub_debate_budget * 0.3))
            final_resolution_prompt = self.tokenizer.trim_text_to_tokens(final_resolution_prompt_base, sub_debate_synthesis_budget)

            synthesizer_persona = self.all_personas.get("Impartial_Arbitrator") or self.all_personas.get("General_Synthesizer")
            if synthesizer_persona:
                try:
                    final_resolution = self._execute_llm_turn(
                        synthesizer_persona.name, synthesizer_persona, final_resolution_prompt, "sub_debate_synthesis", synthesizer_persona.max_tokens
                    )
                    self._log_with_context("info", "Conflict successfully resolved.", resolution_summary=final_resolution)
                    return {"conflict_resolved": True, "resolution_summary": final_resolution}
                except Exception as e:
                    self._log_with_context("error", f"Failed to synthesize conflict resolution: {e}", exc_info=True)
                    self.rich_console.print(f"[red]Failed to synthesize conflict resolution: {e}[/red]")

        self._log_with_context("warning", "Conflict could not be resolved in sub-debate.")
        return None

    def _summarize_metrics_for_llm(self, metrics: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
        """
        Intelligently summarizes the metrics dictionary to fit within a token budget.
        Prioritizes high-level summaries and truncates verbose lists like 'detailed_issues'.
        Ensures critical configuration and deployment analysis are preserved.
        """
        summarized_metrics = json.loads(json.dumps(metrics)) # Deep copy

        current_tokens = self.tokenizer.count_tokens(json.dumps(summarized_metrics))
        
        if current_tokens <= max_tokens:
            return summarized_metrics
        
        self._log_with_context("debug", f"Summarizing metrics for LLM. Current tokens: {current_tokens}, Max: {max_tokens}")

        # --- Prioritize critical sections: configuration_analysis and deployment_robustness ---
        CRITICAL_SECTION_TOKEN_BUDGET = int(max_tokens * 0.15) # <-- FURTHER REDUCED
        CRITICAL_SECTION_TOKEN_BUDGET = max(50, min(CRITICAL_SECTION_TOKEN_BUDGET, max_tokens * 0.2)) # Cap at 20%
        
        critical_sections_content = {}
        if 'configuration_analysis' in summarized_metrics:
            critical_sections_content['configuration_analysis'] = summarized_metrics['configuration_analysis']
            del summarized_metrics['configuration_analysis']
        if 'deployment_robustness' in summarized_metrics:
            critical_sections_content['deployment_robustness'] = summarized_metrics['deployment_robustness']
            del summarized_metrics['deployment_robustness']

        summarized_critical_sections = {}
        critical_sections_tokens = self.tokenizer.count_tokens(json.dumps(critical_sections_content))
        
        if critical_sections_tokens > CRITICAL_SECTION_TOKEN_BUDGET:
            self._log_with_context("debug", f"Critical sections (config/deployment) exceed dedicated budget. Summarizing aggressively.")
            for key, content in critical_sections_content.items():
                if key == 'configuration_analysis' and content:
                    summarized_config = {
                        "ci_workflow_present": bool(content.get("ci_workflow")),
                        "pre_commit_hooks_count": len(content.get("pre_commit_hooks", [])),
                        "pyproject_toml_present": bool(content.get("pyproject_toml")),
                    }
                    summarized_critical_sections[key] = summarized_config
                elif key == 'deployment_robustness' and content:
                    summarized_deployment = {
                        "dockerfile_present": content.get("dockerfile_present"),
                        "dockerfile_healthcheck_present": content.get("dockerfile_healthcheck_present"),
                        "dockerfile_non_root_user": content.get("dockerfile_non_root_user"),
                        "prod_dependency_count": content.get("prod_dependency_count"),
                        "unpinned_prod_dependencies_count": len(content.get("unpinned_prod_dependencies", [])),
                    }
                    summarized_critical_sections[key] = summarized_deployment
                else:
                    summarized_critical_sections[key] = f"High-level summary of {key} (truncated)."
        else:
            summarized_critical_sections = critical_sections_content

        summarized_metrics.update(summarized_critical_sections)
        current_tokens = self.tokenizer.count_tokens(json.dumps(summarized_metrics))
        remaining_budget_for_issues = max_tokens - current_tokens

        for issue_list_key in ['detailed_issues', 'ruff_violations']:
            if 'code_quality' in summarized_metrics and issue_list_key in summarized_metrics['code_quality'] and summarized_metrics['code_quality'][issue_list_key]:
                original_issues = summarized_metrics['code_quality'][issue_list_key]
                
                # HARD CAP: Only include top 1 issue, or replace with summary
                num_issues_to_keep = min(int(remaining_budget_for_issues / 50), 1) # Assume ~50 tokens per issue, keep max 1
                num_issues_to_keep = max(0, min(len(original_issues), num_issues_to_keep))

                if num_issues_to_keep < len(original_issues):
                    if num_issues_to_keep > 0:
                        summarized_metrics['code_quality'][issue_list_key] = original_issues[:num_issues_to_keep]
                        summarized_metrics['code_quality'][issue_list_key].append({
                            "type": "TRUNCATION_SUMMARY",
                            "message": f"Only top {num_issues_to_keep} {issue_list_key} are listed due to token limits. Total: {len(original_issues)}."
                        })
                    else:
                        summarized_metrics['code_quality'][issue_list_key] = f"Too many {issue_list_key} to list ({len(original_issues)} total). Only high-level counts are provided."
                    self._log_with_context("debug", f"Truncated {issue_list_key} from {len(original_issues)} to {num_issues_to_keep}.")
                    current_tokens = self.tokenizer.count_tokens(json.dumps(summarized_metrics))
                    remaining_budget_for_issues = max_tokens - current_tokens
                elif num_issues_to_keep == 0 and len(original_issues) > 0:
                    del summarized_metrics['code_quality'][issue_list_key]
                    self._log_with_context("debug", f"Removed {issue_list_key} entirely due to lack of budget.")
                    current_tokens = self.tokenizer.count_tokens(json.dumps(summarized_metrics))
                    remaining_budget_for_issues = max_tokens - current_tokens
            elif 'code_quality' in summarized_metrics and issue_list_key in summarized_metrics['code_quality'] and not summarized_metrics['code_quality'][issue_list_key]:
                del summarized_metrics['code_quality'][issue_list_key]
        
        if self.tokenizer.count_tokens(json.dumps(summarized_metrics)) > max_tokens:
            self._log_with_context("warning", "Metrics still too large after aggressive truncation. Converting to high-level summary string.")
            summary_str = f"Code Quality: Ruff: {metrics['code_quality']['ruff_issues_count']}, Smells: {metrics['code_quality']['code_smells_count']}. " \
                          f"Security: Bandit: {metrics['security']['bandit_issues_count']}, AST: {metrics['security']['ast_security_issues_count']}. " \
                          f"Tokens: {metrics['performance_efficiency']['token_usage_stats']['total_tokens']}, Cost: ${metrics['performance_efficiency']['token_usage_stats']['total_cost_usd']:.4f}. " \
                          f"Robustness: Schema failures: {metrics['robustness']['schema_validation_failures_count']}."
            trimmed_summary_str = self.tokenizer.trim_text_to_tokens(summary_str, max(100, int(max_tokens * 0.5)))
            return {"summary_string": trimmed_summary_str}
          
        return summarized_metrics

    def _summarize_debate_history_for_llm(self, debate_history: List[Dict[str, Any]], max_tokens: int) -> List[Dict[str, Any]]:
        """
        Summarizes the debate history to fit within a token budget.
        Prioritizes recent turns and concise summaries of each turn's output.
        """
        summarized_history = []
        current_tokens = 0

        MAX_TURNS_TO_INCLUDE = 1 # <-- FURTHER REDUCED
        
        for turn in reversed(debate_history[-MAX_TURNS_TO_INCLUDE:]):
            turn_copy = json.loads(json.dumps(turn))

            if 'output' in turn_copy and isinstance(turn_copy['output'], dict):
                if 'CRITIQUE_SUMMARY' in turn_copy['output']:
                    turn_copy['output'] = {'CRITIQUE_SUMMARY': turn_copy['output']['CRITIQUE_SUMMARY'][:50] + "..." if len(turn_copy['output']['CRITIQUE_SUMMARY']) > 50 else turn_copy['output']['CRITIQUE_SUMMARY']}
                elif 'summary' in turn_copy['output']:
                    turn_copy['output'] = {'summary': turn_copy['output']['summary'][:50] + "..." if len(turn_copy['output']['summary']) > 50 else turn_copy['output']['summary']}
                elif 'general_output' in turn_copy['output']:
                    turn_copy['output'] = {'general_output': turn_copy['output']['general_output'][:30] + "..." if len(turn_copy['output']['general_output']) > 30 else turn_copy['output']['general_output']}
                elif 'analysisTitle' in turn_copy['output']:
                    turn_copy['output'] = {'analysisTitle': turn_copy['output']['analysisTitle'][:50] + "..."}
                elif 'architectural_analysis' in turn_copy['output']:
                    turn_copy['output'] = {'overall_assessment': turn_copy['output']['architectural_analysis'].get('assessment', 'Architectural analysis performed.')[:50] + "..."}
                elif 'CRITIQUE_POINTS' in turn_copy['output']:
                    turn_copy['output'] = {'CRITIQUE_SUMMARY': turn_copy['output'].get('CRITIQUE_SUMMARY', 'Critique provided.')[:50] + "..."}
                else:
                    turn_copy['output'] = {k: str(v)[:20] for k, v in list(turn_copy['output'].items())[:1]}
            elif 'output' in turn_copy and isinstance(turn_copy['output'], str):
                turn_copy['output'] = turn_copy['output'][:30] + "..." if len(turn_copy['output']) > 30 else turn_copy['output']

            turn_tokens = self.tokenizer.count_tokens(json.dumps(turn_copy))

            if current_tokens + turn_tokens <= max_tokens:
                summarized_history.insert(0, turn_copy)
                current_tokens += turn_tokens
            else:
                self._log_with_context("debug", f"Stopped summarizing debate history due to token limit. Included {len(summarized_history)} turns.")
                break

        if not summarized_history and debate_history:
            self._log_with_context("warning", "Debate history too large, providing minimal summary.")
            return [{"summary": f"Debate history contains {len(debate_history)} turns. Too verbose to include in full."}]

        return summarized_history

    def _perform_synthesis_persona_turn(self, persona_sequence: List[str], debate_persona_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes the final synthesis persona turn based on the debate history.
        """
        self._log_with_context("info", "Executing final synthesis persona turn.")

        synthesis_persona_name = ""
        if self.is_self_analysis:
            synthesis_persona_name = "Self_Improvement_Analyst"
        elif self.domain == "Software Engineering":
            synthesis_persona_name = "Impartial_Arbitrator"
        else:
            synthesis_persona_name = "General_Synthesizer"

        synthesis_persona_config = self.all_personas.get(synthesis_persona_name)
        if not synthesis_persona_config:
            self._log_with_context("error", f"Synthesis persona '{synthesis_persona_name}' configuration not found. Falling back to General_Synthesizer.")
            synthesis_persona_name = "General_Synthesizer"
            synthesis_persona_config = self.all_personas.get(synthesis_persona_name)
            if not synthesis_persona_config:
                raise ChimeraError("No synthesis persona configuration found.")

        synthesis_prompt_parts = [f"Initial Problem: {self.initial_prompt}\n\n"]

        debate_history_summary_budget = int(self.phase_budgets["synthesis"] * 0.1) # e.g., 10% for history (even more aggressive)
        summarized_debate_history = self._summarize_debate_history_for_llm(debate_persona_results, debate_history_summary_budget)
        synthesis_prompt_parts.append(f"Debate History:\n{json.dumps(summarized_debate_history, indent=2)}\n\n")

        if self.intermediate_steps.get("Conflict_Resolution_Attempt"):
            synthesis_prompt_parts.append(f"Conflict Resolution Summary: {json.dumps(self.intermediate_steps['Conflict_Resolution_Attempt']['resolution_summary'], indent=2)}\n\n")
        elif self.intermediate_steps.get("Unresolved_Conflict"):
            synthesis_prompt_parts.append(f"Unresolved Conflict: {json.dumps(self.intermediate_steps['Unresolved_Conflict'], indent=2)}\n\n")

        if synthesis_persona_name == "Self_Improvement_Analyst":
            metrics_collector = ImprovementMetricsCollector(
                initial_prompt=self.initial_prompt,
                debate_history=debate_persona_results,
                intermediate_steps=self.intermediate_steps,
                codebase_context=self.codebase_context,
                tokenizer=self.tokenizer,
                llm_provider=self.llm_provider,
                persona_manager=self.persona_manager,
                content_validator=self.content_validator
            )
            collected_metrics = metrics_collector.collect_all_metrics()
            self.intermediate_steps["Self_Improvement_Metrics"] = collected_metrics

            metrics_summary_budget = int(self.phase_budgets["synthesis"] * 0.2) # e.g., 20% for metrics (more aggressive)
            summarized_metrics = self._summarize_metrics_for_llm(collected_metrics, metrics_summary_budget)
            synthesis_prompt_parts.append(f"Objective Metrics and Analysis:\n{json.dumps(summarized_metrics, indent=2)}\n\n")

            synthesis_prompt_parts.append(
                "Based on the debate history, conflict resolution (if any), and objective metrics, "
                "critically analyze Project Chimera's codebase for self-improvement. "
                "Identify the most impactful code changes for self-improvement, focusing on the 80/20 Pareto principle. "
                "Prioritize enhancements to reasoning quality, robustness, efficiency, and developer maintainability. "
                "For each suggestion, provide a clear rationale and a specific, actionable code modification. "
                "Your output MUST strictly adhere to the SelfImprovementAnalysisOutputV1 JSON schema."
            )
            self.PERSONA_OUTPUT_SCHEMAS["Self_Improvement_Analyst"] = SelfImprovementAnalysisOutputV1

        else:
            synthesis_prompt_parts.append(
                "Based on the initial problem and the debate history, synthesize a final, comprehensive answer. "
                "Address all aspects of the initial problem and integrate insights from all personas. "
                "Your output MUST strictly adhere to the LLMOutput JSON schema."
            )
            self.PERSONA_OUTPUT_SCHEMAS["Impartial_Arbitrator"] = LLMOutput
            self.PERSONA_OUTPUT_SCHEMAS["General_Synthesizer"] = GeneralOutput


        final_synthesis_prompt_raw = "\n".join(synthesis_prompt_parts)
        
        input_budget_for_synthesis_prompt = int(self.phase_budgets["synthesis"] * 0.3) # e.g., 30% for input (more aggressive)
        
        final_synthesis_prompt = self.tokenizer.trim_text_to_tokens(
            final_synthesis_prompt_raw,
            input_budget_for_synthesis_prompt,
            truncation_indicator="\n... (truncated for token limits) ..."
        )

        max_output_tokens_for_turn = self.phase_budgets["synthesis"] - self.tokenizer.count_tokens(final_synthesis_prompt)
        max_output_tokens_for_turn = max(500, max_output_tokens_for_turn)
        
        estimated_tokens_for_turn = self.tokenizer.count_tokens(final_synthesis_prompt) + max_output_tokens_for_turn
        self.check_budget("synthesis", estimated_tokens_for_turn, synthesis_persona_name)

        synthesis_output = self._execute_llm_turn(
            synthesis_persona_name,
            synthesis_persona_config,
            final_synthesis_prompt,
            "synthesis",
            max_output_tokens_for_turn
        )
        self._log_with_context("info", "Final synthesis persona turn completed.")
        return synthesis_output

    def _finalize_debate_results(self, context_persona_turn_results: Optional[Dict[str, Any]], debate_persona_results: List[Dict[str, Any]], synthesis_persona_results: Optional[Dict[str, Any]]) -> Tuple[Any, Dict[str, Any]]:
        """Synthesizes the final answer and prepares the intermediate steps for display."""
        final_answer = synthesis_persona_results

        if not isinstance(final_answer, dict):
            final_answer = {"general_output": str(final_answer), "malformed_blocks": []}
        if "malformed_blocks" not in final_answer:
            final_answer["malformed_blocks"] = []

        if isinstance(final_answer, dict):
            if final_answer.get("version") == "1.0" and "data" in final_answer:
                final_answer["data"] = self._consolidate_self_improvement_code_changes(final_answer["data"])
                self._log_with_context("info", "Self-improvement code changes consolidated (versioned output).")
            elif "ANALYSIS_SUMMARY" in final_answer and "IMPACTFUL_SUGGESTIONS" in final_answer:
                v1_data_consolidated = self._consolidate_self_improvement_code_changes(final_answer)
                final_answer = SelfImprovementAnalysisOutput(
                    version="1.0",
                    data=v1_data_consolidated,
                    malformed_blocks=final_answer.get("malformed_blocks", [])
                ).model_dump(by_alias=True)
                self._log_with_context("info", "Self-improvement code changes consolidated and wrapped (direct V1 output).")


        self._update_intermediate_steps_with_totals()
        if "malformed_blocks" not in self.intermediate_steps:
            self.intermediate_steps["malformed_blocks"] = []

        if self.intermediate_steps.get("Conflict_Resolution_Attempt"):
            final_answer["CONFLICT_RESOLUTION"] = self.intermediate_steps["Conflict_Resolution_Attempt"]["resolution_summary"]
            final_answer["UNRESOLVED_CONFLICT"] = None
        elif self.intermediate_steps.get("Unresolved_Conflict"):
            final_answer["UNRESOLVED_CONFLICT"] = self.intermediate_steps["Unresolved_Conflict"]["summary"]
            final_answer["CONFLICT_RESOLUTION"] = None

        return final_answer, self.intermediate_steps

    def _update_intermediate_steps_with_totals(self):
        """Updates the intermediate steps dictionary with total token usage and estimated cost."""
        self.intermediate_steps["Total_Tokens_Used"] = self.tokens_used
        self.intermediate_steps["Total_Estimated_Cost_USD"] = self.get_total_estimated_cost()

    @handle_errors(default_return=None, log_level="ERROR")
    def run_debate(self) -> Tuple[Any, Dict[str, Any]]:
        """
        Orchestrates the full Socratic debate process.
        Returns the final synthesized answer and a dictionary of intermediate steps.
        """
        self._initialize_debate_state()

        initial_persona_sequence = self._get_final_persona_sequence(self.initial_prompt, None)
        self.intermediate_steps["Persona_Sequence_Initial"] = initial_persona_sequence

        self.status_callback(message="Phase 1: Analyzing Context...", state="running",
                             current_total_tokens=self.tokens_used, current_total_cost=self.get_total_estimated_cost(),
                             progress_pct=self.get_progress_pct("context"))
        context_analysis_results = self._perform_context_analysis(tuple(initial_persona_sequence))
        self.intermediate_steps["Context_Analysis_Output"] = context_analysis_results

        persona_sequence = self._get_final_persona_sequence(self.initial_prompt, context_analysis_results)
        self.intermediate_steps["Persona_Sequence"] = persona_sequence

        self._distribute_debate_persona_budgets(persona_sequence)

        context_persona_turn_results = None
        if "Context_Aware_Assistant" in persona_sequence:
            self.status_callback(message="Phase 2: Context-Aware Assistant Turn...", state="running",
                                 current_total_tokens=self.tokens_used, current_total_cost=self.get_total_estimated_cost(),
                                 progress_pct=self.get_progress_pct("debate"), current_persona_name="Context_Aware_Assistant")
            context_persona_turn_results = self._process_context_persona_turn(persona_sequence, context_analysis_results)
            self.intermediate_steps["Context_Aware_Assistant_Output"] = context_persona_turn_results

        self.status_callback(message="Phase 3: Executing Debate Turns...", state="running",
                             current_total_tokens=self.tokens_used, current_total_cost=self.get_total_estimated_cost(),
                             progress_pct=self.get_progress_pct("debate"))
        debate_persona_results = self._execute_debate_persona_turns(persona_sequence, context_persona_turn_results)
        self.intermediate_steps["Debate_History"] = debate_persona_results

        self.status_callback(message="Phase 4: Synthesizing Final Answer...", state="running",
                             current_total_tokens=self.tokens_used, current_total_cost=self.get_total_estimated_cost(),
                             progress_pct=self.get_progress_pct("synthesis"))

        synthesis_persona_results = self._perform_synthesis_persona_turn(persona_sequence, debate_persona_results)
        self.intermediate_steps["Final_Synthesis_Output"] = synthesis_persona_results

        self.status_callback(message="Finalizing Results...", state="running",
                             current_total_tokens=self.tokens_used, current_total_cost=self.get_total_estimated_cost(),
                             progress_pct=0.95)
        final_answer, intermediate_steps = self._finalize_debate_results(context_persona_turn_results, debate_persona_results, synthesis_persona_results)

        self.status_callback(message="Socratic Debate Complete!", state="complete",
                             current_total_tokens=self.tokens_used, current_total_cost=self.get_total_estimated_cost(),
                             progress_pct=1.0)
        self._log_with_context("info", "Socratic Debate process completed successfully.",
                               total_tokens=self.tokens_used, total_cost=self.get_total_estimated_cost())

        return final_answer, intermediate_steps

    def _consolidate_self_improvement_code_changes(self, analysis_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Consolidates multiple CODE_CHANGES_SUGGESTED for the same file within
        SelfImprovementAnalysisOutput. Also filters out no-op changes.
        """
        if "IMPACTFUL_SUGGESTIONS" not in analysis_output:
            return analysis_output

        consolidated_suggestions = []
        for suggestion in analysis_output["IMPACTFUL_SUGGESTIONS"]:
            if "CODE_CHANGES_SUGGESTED" not in suggestion or not suggestion["CODE_CHANGES_SUGGESTED"]:
                consolidated_suggestions.append(suggestion)
                continue

            file_changes_map = defaultdict(list)
            for change_data in suggestion["CODE_CHANGES_SUGGESTED"]:
                try:
                    code_change = CodeChange.model_validate(change_data)
                    file_changes_map[code_change.file_path].append(code_change)
                except ValidationError as e:
                    self._log_with_context("warning", f"Malformed CodeChange item during consolidation: {e}. Skipping.",
                                           raw_change_data=change_data)
                    analysis_output.setdefault("malformed_blocks", []).append({
                        "type": "CODE_CHANGE_CONSOLIDATION_ERROR",
                        "message": f"Malformed CodeChange item skipped during consolidation: {e}",
                        "raw_string_snippet": str(change_data)[:500]
                    })
                    continue

            new_code_changes_for_suggestion = []
            for file_path, changes_for_file in file_changes_map.items():
                remove_actions = [c for c in changes_for_file if c.action == "REMOVE"]
                add_actions = [c for c in changes_for_file if c.action == "ADD"]
                modify_actions = [c for c in changes_for_file if c.action == "MODIFY"]

                if remove_actions:
                    all_lines_to_remove = []
                    for ra in remove_actions:
                        all_lines_to_remove.extend(ra.lines)
                    new_code_changes_for_suggestion.append(CodeChange(
                        FILE_PATH=file_path,
                        ACTION="REMOVE",
                        LINES=list(set(all_lines_to_remove))
                    ).model_dump(by_alias=True))
                    self._log_with_context("info", f"Consolidated multiple changes for {file_path} into a single REMOVE action.")
                elif add_actions:
                    new_code_changes_for_suggestion.append(add_actions[0].model_dump(by_alias=True))
                    self._log_with_context("info", f"Consolidated multiple changes for {file_path} into a single ADD action.")
                elif modify_actions:
                    original_content = self.codebase_context.get(file_path, "")
                    
                    final_content_for_diff = original_content
                    last_full_content_provided = None
                    
                    for mod_change in modify_actions:
                        if mod_change.full_content is not None:
                            last_full_content_provided = mod_change.full_content
                    
                    consolidated_diff_content = None

                    if last_full_content_provided is not None:
                        final_content_for_diff = last_full_content_provided
                        # Generate unified diff
                        diff_lines = difflib.unified_diff(
                            original_content.splitlines(keepends=True),
                            final_content_for_diff.splitlines(keepends=True),
                            fromfile=f"a/{file_path}",
                            tofile=f"b/{file_path}",
                            lineterm=''
                        )
                        consolidated_diff_content = "".join(diff_lines)
                        self._log_with_context("debug", f"Generated diff from FULL_CONTENT for {file_path}.")
                    else:
                        last_diff_from_llm = None
                        for mod_change in modify_actions:
                            if mod_change.diff_content is not None:
                                last_diff_from_llm = mod_change.diff_content
                        
                        if last_diff_from_llm is not None:
                            consolidated_diff_content = last_diff_from_llm
                            self._log_with_context("debug", f"Using LLM-provided DIFF_CONTENT for {file_path}.")
                        else:
                            self._log_with_context("info", f"Consolidated MODIFY for {file_path} resulted in no effective change (no FULL_CONTENT or DIFF_CONTENT provided). Removing from suggestions.")
                            analysis_output.setdefault("malformed_blocks", []).append({
                                "type": "NO_OP_CODE_CHANGE_CONSOLIDATED",
                                "message": f"Consolidated MODIFY for {file_path} resulted in no effective change. Removed from final suggestions.",
                                "file_path": file_path,
                                "suggestion_area": suggestion.get("AREA")
                            })
                            continue
                    
                    if consolidated_diff_content and consolidated_diff_content.strip():
                        new_code_changes_for_suggestion.append(CodeChange(
                            FILE_PATH=file_path,
                            ACTION="MODIFY",
                            DIFF_CONTENT=consolidated_diff_content
                        ).model_dump(by_alias=True))
                        self._log_with_context("info", f"Consolidated multiple MODIFY actions for {file_path} into a single DIFF_CONTENT.")
                    else:
                        self._log_with_context("info", f"Consolidated MODIFY for {file_path} resulted in no effective change. Removing from suggestions.")
                        analysis_output.setdefault("malformed_blocks", []).append({
                            "type": "NO_OP_CODE_CHANGE_CONSOLIDATED",
                            "message": f"Consolidated MODIFY for {file_path} resulted in no effective change. Removed from final suggestions.",
                            "file_path": file_path,
                            "suggestion_area": suggestion.get("AREA")
                        })

            suggestion["CODE_CHANGES_SUGGESTED"] = new_code_changes_for_suggestion
            consolidated_suggestions.append(suggestion)

        analysis_output["IMPACTFUL_SUGGESTIONS"] = consolidated_suggestions
        return analysis_output