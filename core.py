# core.py
import copy
import difflib
import gc
import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime
from functools import lru_cache
from typing import Any, Callable, Optional, Union

from pydantic import BaseModel, ValidationError
from rich.console import Console

from src.config.model_registry import ModelRegistry
from src.config.settings import ChimeraSettings

# Import the global settings instance
from src.config.settings import settings as global_settings_instance
from src.conflict_resolution import ConflictResolutionManager
from src.constants import CRITICAL_FILES_FOR_SELF_ANALYSIS, SHARED_JSON_INSTRUCTIONS

# --- IMPORT MODIFICATIONS ---
# NEW IMPORT FOR CODEBASE SCANNING
from src.context.context_analyzer import CodebaseScanner, ContextRelevanceAnalyzer

# NEW: Import monitoring components
# --- NEW: Separate modules for complex logic ---
from src.exceptions import (
    ChimeraError,
    CircuitBreakerError,
    CodebaseAccessError,
    LLMProviderError,
    LLMProviderRequestError,
    LLMProviderResponseError,
    SchemaValidationError,
    TokenBudgetExceededError,
)

# NEW IMPORT: For centralized LLM calls
from src.llm.orchestrator import LLMOrchestrator
from src.llm_provider import GeminiProvider
from src.models import (
    CodeChange,
    PersonaConfig,
    SelfImprovementAnalysisOutput,
    SelfImprovementAnalysisOutputV1,
    SuggestionItem,  # Added for _handle_codebase_access_error
)
from src.monitoring.performance_logger import get_performance_logger
from src.monitoring.system_monitor import MetricType, get_system_monitor
from src.persona.routing import PersonaRouter
from src.persona_manager import PersonaManager
from src.rag_system import KnowledgeRetriever, RagOrchestrator  # New Import for RAG

# NEW IMPORTS FOR SELF-IMPROVEMENT
from src.self_improvement import (
    ContentAlignmentValidator,
    CritiqueEngine,
    FocusedMetricsCollector,
    ImprovementApplicator,
    StrategyManager,
)

# NEW: Import from modularized self_improvement package
from src.token_tracker import TokenUsageTracker

# from src.logging_config import setup_structured_logging # Removed as logging is global
from src.utils.core_helpers.error_handler import handle_errors  # Updated import
from src.utils.core_helpers.json_utils import (
    convert_to_json_friendly,
    safe_json_dumps,
    safe_json_loads,
)

# MODIFIED: Import safe_json_loads and safe_json_dumps
# NEW IMPORT for PromptOptimizer
from src.utils.prompting.prompt_optimizer import PromptOptimizer  # Updated import
from src.utils.reporting.output_parser import LLMOutputParser  # Updated import
from src.utils.validation.json_validator import validate_llm_output  # NEW

# --- CONSTANTS FOR PLR2004 FIXES ---
COMPLEXITY_HIGH_THRESHOLD = 0.7
MIN_DEBATE_RATIO = 0.05
RATIO_NORMALIZATION_TOLERANCE = 1e-6
# --- END CONSTANTS ---

logger = logging.getLogger(__name__)


class SocraticDebate:
    def __init__(
        self,
        initial_prompt: str,
        api_key: str,
        structured_codebase_context: Optional[dict[str, Any]] = None,
        raw_file_contents: Optional[dict[str, str]] = None,
        settings: Optional[ChimeraSettings] = None,
        all_personas: Optional[dict[str, PersonaConfig]] = None,
        persona_sets: Optional[dict[str, list[str]]] = None,
        domain: Optional[str] = None,
        model_name: str = "gemini-2.5-flash-lite",
        status_callback: Optional[Callable] = None,
        rich_console: Optional[Console] = None,
        context_analyzer: Optional[ContextRelevanceAnalyzer] = None,
        is_self_analysis: bool = False,
        persona_manager: Optional[PersonaManager] = None,
        content_validator: Optional[ContentAlignmentValidator] = None,
        # NEW: Add explicit optional dependencies for full injection
        prompt_optimizer: Optional[PromptOptimizer] = None,
        critique_engine: Optional[CritiqueEngine] = None,
        improvement_applicator: Optional[ImprovementApplicator] = None,
        conflict_resolution_manager: Optional[
            ConflictResolutionManager
        ] = None,  # NEW: Inject ConflictResolutionManager
        model_registry: Optional[ModelRegistry] = None,  # NEW: Inject ModelRegistry
        token_tracker: Optional[TokenUsageTracker] = None,
        codebase_scanner: Optional[CodebaseScanner] = None,
        summarizer_pipeline_instance: Any = None,
        llm_orchestrator: Optional[LLMOrchestrator] = None,
    ):
        """Initializes the Socratic debate session."""
        # setup_structured_logging(log_level=logging.INFO) # Removed: Logging setup is now handled globally in app.py
        self.logger = logging.getLogger(__name__)

        # MODIFIED: Use the global settings instance as default
        self.settings = settings or global_settings_instance
        self.max_total_tokens_budget = self.settings.total_budget
        self.model_name = model_name
        self.status_callback = status_callback
        self.rich_console = rich_console or Console(stderr=True)
        self.request_id = str(uuid.uuid4())[:8]
        self._log_extra = {"request_id": self.request_id or "N/A"}

        # NEW: Initialize monitoring system
        self.system_monitor = get_system_monitor()
        self.debate_start_time = None

        self.initial_prompt = initial_prompt
        self.intermediate_steps = {}
        self.is_self_analysis = is_self_analysis

        self.codebase_scanner = codebase_scanner
        if self.codebase_scanner:
            self.raw_file_contents = self.codebase_scanner.raw_file_contents
            self.structured_codebase_context = self.codebase_scanner.file_structure
        else:
            self.raw_file_contents = raw_file_contents or {}
            self.structured_codebase_context = structured_codebase_context or {}

        if (
            self.is_self_analysis
            and not self.raw_file_contents
            and self.codebase_scanner
        ):
            self.logger.info(
                "Performing self-analysis - scanning codebase for context (fallback scan in SocraticDebate init)..."
            )
            full_codebase_analysis = self.codebase_scanner.load_own_codebase_context()
            self.structured_codebase_context = full_codebase_analysis.get(
                "file_structure", {}
            )
            self.raw_file_contents = full_codebase_analysis.get("raw_file_contents", {})
            self.logger.info(
                f"Codebase context gathered: {len(self.structured_codebase_context)} directories scanned during fallback."
            )

        if not self.raw_file_contents:
            self.logger.warning(
                "Codebase context is empty after SocraticDebate initialization. Context-aware features may be limited.",
                extra=self._log_extra,
            )

        self.domain = domain

        self.token_tracker = token_tracker or TokenUsageTracker(
            budget=self.max_total_tokens_budget
        )

        # New: Initialize RAG system if codebase is present
        self.knowledge_retriever: Optional[KnowledgeRetriever] = None
        self.rag_orchestrator: Optional[RagOrchestrator] = None
        if self.raw_file_contents:
            self.knowledge_retriever = KnowledgeRetriever(
                raw_file_contents=self.raw_file_contents
            )
            # PersonaManager is needed for RagOrchestrator, but PersonaManager itself needs PromptOptimizer.
            # This creates a circular dependency if PersonaManager is not yet fully initialized.
            # For now, initialize with None and set later.
            self.rag_orchestrator = RagOrchestrator(self.knowledge_retriever, None)

        # Initialize PromptOptimizer first, as it's a dependency for GeminiProvider and PersonaManager
        self.prompt_optimizer = (
            prompt_optimizer
            or PromptOptimizer(  # Use injected or instantiate
                tokenizer=None,  # Will be set after GeminiProvider init
                settings=self.settings,
                summarizer_pipeline=summarizer_pipeline_instance,
                persona_manager=persona_manager,  # Pass persona_manager to PromptOptimizer
                token_tracker=self.token_tracker,
            )
        )

        try:
            self.llm_provider = GeminiProvider(
                api_key=api_key,
                model_name=self.model_name,
                rich_console=self.rich_console,
                request_id=self.request_id,
                settings=self.settings,
                summarizer_pipeline_instance=summarizer_pipeline_instance,
                prompt_optimizer_instance=self.prompt_optimizer,
                token_tracker=self.token_tracker,  # Pass token_tracker to LLMProvider
            )
        except LLMProviderError as e:
            self._log_with_context(
                "error",
                f"Failed to initialize LLM provider: {e.message}",
                exc_info=True,
            )
            raise ChimeraError(
                f"LLM provider initialization failed: {e.message}", original_exception=e
            ) from e
        except Exception as e:
            self._log_with_context(
                "error",
                f"An unexpected error occurred during LLM provider initialization: {e}",
                exc_info=True,
            )
            raise ChimeraError(
                f"LLM provider initialization failed unexpectedly: {e}",
                original_exception=e,
            ) from e

        try:
            self.tokenizer = self.llm_provider.tokenizer
            # Now that tokenizer is available, set it in prompt_optimizer
            self.prompt_optimizer.tokenizer = self.tokenizer
        except AttributeError as e:
            raise ChimeraError(
                "LLM provider tokenizer is not available."
            ) from e  # FIX B904

        self.persona_manager = persona_manager
        if not self.persona_manager:
            self.logger.warning(
                "PersonaManager instance not provided to SocraticDebate. Initializing a new one. This might affect state persistence in UI."
            )
            self.persona_manager = PersonaManager(
                self.settings.domain_keywords,
                token_tracker=self.token_tracker,
                settings=self.settings,
                prompt_optimizer=self.prompt_optimizer,
            )
        self.persona_manager.token_tracker = self.token_tracker
        self.persona_manager.settings = self.settings
        self.persona_manager.prompt_optimizer = self.prompt_optimizer

        # If RAG Orchestrator was initialized without persona_manager, set it now
        if self.rag_orchestrator and self.rag_orchestrator.persona_manager is None:
            self.rag_orchestrator.persona_manager = self.persona_manager

        self.all_personas = self.persona_manager.all_personas
        self.persona_sets = self.persona_manager.persona_sets

        self.persona_router = self.persona_manager.persona_router
        if not self.persona_router:
            self.logger.warning(
                "PersonaRouter not found in PersonaManager. Initializing a new one. This might indicate an issue in PersonaManager setup."
            )
            self.persona_router = PersonaRouter(
                self.all_personas,
                self.persona_sets,
                self.persona_manager.prompt_analyzer,
            )

        self.context_analyzer = context_analyzer
        if not self.context_analyzer:
            self.logger.warning(
                "ContextRelevanceAnalyzer instance not provided. Initializing a new one."
            )
            self.context_analyzer = ContextRelevanceAnalyzer(
                cache_dir=self.settings.sentence_transformer_cache_dir,
                raw_file_contents=self.raw_file_contents,
                summarizer_pipeline=summarizer_pipeline_instance,  # ADDED
            )
            if self.persona_router:
                self.context_analyzer.set_persona_router(self.persona_router)
        else:
            # Ensure the existing context_analyzer is updated with the summarizer_pipeline_instance
            # if it was not passed during its own initialization.
            # This is a defensive measure. Ideally, it should be passed consistently.
            if (
                not hasattr(self.context_analyzer, "summarizer_pipeline")
                or self.context_analyzer.summarizer_pipeline is None
            ):
                self.context_analyzer.summarizer_pipeline = summarizer_pipeline_instance
            self.context_analyzer.raw_file_contents = self.raw_file_contents
            current_files_hash = hash(frozenset(self.raw_file_contents.items()))
            if (
                not hasattr(self.context_analyzer, "_last_raw_file_contents_hash")
                or self.context_analyzer._last_raw_file_contents_hash
                != current_files_hash
            ):
                try:
                    self.context_analyzer.compute_file_embeddings(
                        self.raw_file_contents
                    )
                    self.context_analyzer._last_raw_file_contents_hash = (
                        current_files_hash
                    )
                    self._log_with_context(
                        "info",
                        "Computed file embeddings for codebase context during SocraticDebate init.",
                    )
                except Exception as e:
                    self._log_with_context(
                        "error", f"Error computing context embeddings: {e}[/red]"
                    )
                    if self.status_callback:
                        self.status_callback(
                            f"[red]Error computing context embeddings: {e}[/red]",
                            state="error",
                            current_total_tokens=self.token_tracker.current_usage,
                            current_total_cost=self.get_total_estimated_cost(),
                            progress_pct=self.get_progress_pct("context", error=True),
                            current_persona_name="Context_Relevance_Analyzer_Init",
                        )
                    raise ChimeraError(
                        f"Error computing context embeddings: {e}", original_exception=e
                    ) from e

        self.content_validator = content_validator or ContentAlignmentValidator(
            original_prompt=self.initial_prompt, debate_domain=self.domain
        )

        # NEW: Initialize StrategyManager, CritiqueEngine, ImprovementApplicator if not already passed
        self.strategy_manager = StrategyManager()
        self.critique_engine = (
            critique_engine or CritiqueEngine()
        )  # Use injected or instantiate
        self.improvement_applicator = (
            improvement_applicator or ImprovementApplicator()
        )  # Use injected or instantiate

        self._calculate_token_budgets()
        self.conflict_manager = (
            conflict_resolution_manager
            or ConflictResolutionManager(  # Use injected or instantiate
                llm_provider=self.llm_provider, persona_manager=self.persona_manager
            )
        )
        self.model_registry = (
            model_registry or ModelRegistry()
        )  # Use injected or instantiate

        self.model_name = self._determine_optimal_model(model_name)

        # Initialize LLMOrchestrator
        self.llm_orchestrator = llm_orchestrator or LLMOrchestrator(
            llm_provider=self.llm_provider,
            token_tracker=self.token_tracker,
            settings=self.settings,
            request_id=self.request_id,
        )

        self.synthesis_persona_name_for_metrics: Optional[str] = None
        self.file_analysis_cache: dict[str, dict[str, Any]] = {}
        self.output_parser = LLMOutputParser()
        self._cleanup_performed = False

    def close(self):
        """Explicitly cleans up resources held by the SocraticDebate instance."""
        if self._cleanup_performed:
            self._log_with_context("debug", "SocraticDebate cleanup already performed.")
            return

        self._log_with_context("info", "Cleaning up SocraticDebate resources...")

        if hasattr(self, "llm_provider") and self.llm_provider:
            try:
                if hasattr(self.llm_provider, "close") and callable(
                    self.llm_provider.close
                ):
                    self.llm_provider.close()
                    self._log_with_context("debug", "LLM Provider resources closed.")
                else:
                    self._log_with_context(
                        "warning", "LLM Provider does not have a close() method."
                    )
            except Exception as e:
                self._log_with_context(
                    "error", f"Error closing LLM Provider: {e}", exc_info=True
                )

        if hasattr(self, "llm_orchestrator") and self.llm_orchestrator:
            try:
                if hasattr(self.llm_orchestrator, "close") and callable(
                    self.llm_orchestrator.close
                ):
                    self.llm_orchestrator.close()
                    self._log_with_context(
                        "debug", "LLM Orchestrator resources closed."
                    )
            except Exception as e:
                self._log_with_context(
                    "error", f"Error closing LLM Orchestrator: {e}", exc_info=True
                )

        if hasattr(self, "file_analysis_cache"):
            self.file_analysis_cache = {}
        if hasattr(self, "intermediate_steps"):
            self.intermediate_steps = {}
        if hasattr(self, "debate_history"):
            self.debate_history = []

        gc.collect()
        self._log_with_context(
            "debug", "Garbage collection triggered during SocraticDebate cleanup."
        )

        self._cleanup_performed = True
        self._log_with_context("info", "SocraticDebate resources cleaned up.")

    def _log_with_context(self, level: str, message: str, **kwargs):
        """Helper to add request context to all logs from this instance using the class-specific logger."""
        exc_info = kwargs.pop("exc_info", None)
        original_exception = kwargs.pop("original_exception", None)
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
            log_data["original_exception_type"] = type(original_exception).__name__
            log_data["original_exception_message"] = str(original_exception)
            logger_method(message, extra=log_data)
        else:
            logger_method(message, extra=log_data)

    def _determine_optimal_model(self, preferred_model_name: str) -> str:
        """Determine the best model to use based on requirements and availability"""
        requirements = ["reasoning"]
        if "coding" in self.domain.lower():
            requirements.append("coding")

        if not hasattr(self, "model_registry") or self.model_registry is None:
            self.model_registry = ModelRegistry()

        model = self.model_registry.get_model(
            requirements=requirements,
            budget=None,
            preferred_model_name=preferred_model_name,
        )
        return model.name if model else preferred_model_name

    def _determine_phase_ratios(
        self, prompt_analysis: dict[str, Any]
    ) -> tuple[float, float, float]:
        """Determines the token budget ratios for context, debate, and synthesis phases."""
        complexity_score = prompt_analysis["complexity_score"]

        if not hasattr(self, "persona_manager") or self.persona_manager is None:
            self.logger.warning("PersonaManager not initialized, using default ratios.")
            return 0.2, 0.7, 0.1

        if self.persona_manager.prompt_analyzer.is_self_analysis_prompt(
            self.initial_prompt
        ):
            context_ratio = self.settings.self_analysis_context_ratio
            debate_ratio = self.settings.self_analysis_debate_ratio
            synthesis_ratio = self.settings.self_analysis_synthesis_ratio
        else:
            base_context_ratio = self.settings.context_token_budget_ratio
            base_output_ratio = self.settings.synthesis_token_budget_ratio

            if (
                "code" in self.initial_prompt.lower()
                or "implement" in self.initial_prompt.lower()
            ):
                context_ratio = max(0.4, base_context_ratio - 0.15)
                synthesis_ratio = min(0.5, base_output_ratio + 0.15)
            elif complexity_score > COMPLEXITY_HIGH_THRESHOLD:  # FIX PLR2004
                context_ratio = min(
                    COMPLEXITY_HIGH_THRESHOLD, base_context_ratio + 0.15
                )  # FIX PLR2004
                synthesis_ratio = max(0.2, base_output_ratio - 0.05)
            else:
                context_ratio = base_context_ratio
                synthesis_ratio = base_output_ratio

            debate_ratio = 1.0 - context_ratio - synthesis_ratio
            if debate_ratio < MIN_DEBATE_RATIO:  # FIX PLR2004
                debate_ratio = MIN_DEBATE_RATIO  # FIX PLR2004
                remaining_for_context_synthesis = 1.0 - debate_ratio
                context_ratio = (
                    context_ratio
                    / (context_ratio + synthesis_ratio)
                    * remaining_for_context_synthesis
                )
                synthesis_ratio = (
                    synthesis_ratio
                    / (context_ratio + synthesis_ratio)
                    * remaining_for_context_synthesis
                )

        total_current_ratios = context_ratio + debate_ratio + synthesis_ratio
        if (
            total_current_ratios > 0
            and abs(total_current_ratios - 1.0) > RATIO_NORMALIZATION_TOLERANCE
        ):  # FIX PLR2004
            self._log_with_context(
                "warning",
                f"Token budget ratios sum to {total_current_ratios}, normalizing them.",
                original_context_ratio=context_ratio,
                original_debate_ratio=debate_ratio,
                original_synthesis_ratio=synthesis_ratio,
            )
            normalization_factor = 1.0 / total_current_ratios
            context_ratio *= normalization_factor
            debate_ratio *= normalization_factor
            synthesis_ratio *= normalization_factor
            self._log_with_context(
                "debug",
                f"Normalized ratios: context={context_ratio}, debate={debate_ratio}, synthesis={synthesis_ratio}",
            )

        return context_ratio, debate_ratio, synthesis_ratio

    def _adjust_and_allocate_budgets(
        self,
        remaining_tokens: int,
        context_ratio: float,
        debate_ratio: float,
        synthesis_ratio: float,
    ) -> dict[str, int]:
        """Allocates and adjusts token budgets for each phase."""
        MIN_PHASE_TOKENS = 250

        context_tokens_budget = int(remaining_tokens * context_ratio)
        debate_tokens_budget = int(remaining_tokens * debate_ratio)
        synthesis_tokens_budget = int(remaining_tokens * synthesis_ratio)

        context_tokens_budget = max(MIN_PHASE_TOKENS, context_tokens_budget)
        debate_tokens_budget = max(MIN_PHASE_TOKENS, debate_tokens_budget)
        synthesis_tokens_budget = max(MIN_PHASE_TOKENS, synthesis_tokens_budget)

        total_allocated = (
            context_tokens_budget + debate_tokens_budget + synthesis_tokens_budget
        )
        if total_allocated > remaining_tokens:
            reduction_factor = 1.0
            if total_allocated > 0:
                reduction_factor = remaining_tokens / total_allocated
            context_tokens_budget = int(context_tokens_budget * reduction_factor)
            debate_tokens_budget = int(debate_tokens_budget * reduction_factor)
            synthesis_tokens_budget = int(synthesis_tokens_budget * reduction_factor)

            context_tokens_budget = max(MIN_PHASE_TOKENS, context_tokens_budget)
            debate_tokens_budget = max(MIN_PHASE_TOKENS, debate_tokens_budget)
            synthesis_tokens_budget = max(MIN_PHASE_TOKENS, synthesis_tokens_budget)

        return {
            "context": context_tokens_budget,
            "debate": debate_tokens_budget,
            "synthesis": synthesis_tokens_budget,
            "persona_turn_budgets": {},
        }

    def _calculate_token_budgets(self):
        """Calculates token budgets for different phases based on context, model limits, and prompt type."""
        try:
            prompt_analysis = self.persona_manager.prompt_analyzer.analyze_complexity(
                self.initial_prompt
            )
            context_ratio, debate_ratio, synthesis_ratio = self._determine_phase_ratios(
                prompt_analysis
            )

            context_str = (
                self.context_analyzer.get_context_summary()
                if self.context_analyzer
                else ""
            )
            self.initial_input_tokens = self.tokenizer.count_tokens(
                context_str + self.initial_prompt
            )

            remaining_tokens = max(
                0, self.max_total_tokens_budget - self.initial_input_tokens
            )
            self.phase_budgets = self._adjust_and_allocate_budgets(
                remaining_tokens, context_ratio, debate_ratio, synthesis_ratio
            )

            self._log_with_context(
                "info",
                "SocraticDebate token budgets initialized",
                initial_input_tokens=self.initial_input_tokens,
                context_budget=self.phase_budgets["context"],
                debate_budget=self.phase_budgets["debate"],
                synthesis_budget=self.phase_budgets["synthesis"],
                max_total_tokens_budget=self.max_total_tokens_budget,
                prompt_complexity=prompt_analysis,
                is_self_analysis=self.is_self_analysis,
            )

        except Exception as e:
            self._log_with_context(
                "error",
                "Token budget calculation failed",
                error=str(e),
                context="token_budget",
                exc_info=True,
                original_exception=e,
            )
            self.phase_budgets = {
                "context": 500,
                "debate": 15000,
                "synthesis": 1000,
                "persona_turn_budgets": {},
            }
            self.initial_input_tokens = 0
            raise ChimeraError(
                "Failed to calculate token budgets due to an unexpected error.",
                original_exception=e,
            ) from e

    def track_token_usage(
        self, phase: str, tokens: int, persona_name: Optional[str] = None
    ):
        """Tracks token usage for a given phase."""
        # This method is now primarily for non-LLM token usage, as LLM calls are tracked by LLMOrchestrator
        self.token_tracker.set_current_stage(phase)
        # The record_usage in TokenUsageTracker now expects prompt_tokens and completion_tokens separately, plus success status
        # For non-LLM usage, we can treat all tokens as 'prompt' tokens for simplicity or adjust as needed.
        self.token_tracker.record_usage(
            tokens, 0, persona=persona_name, is_successful_turn=True
        )
        cost = self.llm_provider.calculate_usd_cost(
            tokens, 0
        )  # This cost calculation is for non-LLM tokens
        self.intermediate_steps.setdefault(f"{phase}_Tokens_Used", 0)
        self.intermediate_steps[f"{phase}_Tokens_Used"] += tokens
        self.intermediate_steps.setdefault(f"{phase}_Estimated_Cost_USD", 0.0)
        self.intermediate_steps[f"{phase}_Estimated_Cost_USD"] += cost
        self._log_with_context(
            "debug",
            f"Tokens used in {phase}: {tokens}. Total: {self.token_tracker.current_usage}",
            phase=phase,
            tokens_added=tokens,
            total_tokens=self.token_tracker.current_usage,
        )

    def check_budget(self, phase: str, tokens_needed: int, step_name: str):
        """Checks if adding tokens for the next step would exceed the budget."""
        if (
            self.token_tracker.current_usage + tokens_needed
            > self.max_total_tokens_budget
        ):
            self._log_with_context(
                "warning",
                f"Token budget exceeded for {step_name} in {phase} phase. Current usage: {self.token_tracker.current_usage}, Needed: {tokens_needed}, Budget: {self.max_total_tokens_budget}",
                current_tokens=self.token_tracker.current_usage,
                tokens_needed=tokens_needed,
                budget=self.max_total_tokens_budget,
                step=step_name,
                phase=phase,
            )
            raise TokenBudgetExceededError(
                self.token_tracker.current_usage,
                self.max_total_tokens_budget,
                details={
                    "phase": phase,
                    "step_name": step_name,
                    "tokens_needed": tokens_needed,
                },
            )

    def get_total_used_tokens(self) -> int:  # Added self
        """Returns the total tokens used so far."""
        return self.token_tracker.current_usage

    def get_total_estimated_cost(self) -> float:
        """Returns the total estimated cost so far."""
        # This should now query the token_tracker for granular usage and calculate cost
        granular_usage = self.token_tracker.get_granular_usage_summary()
        return self.llm_provider.calculate_usd_cost(
            granular_usage["total_prompt_tokens"],
            granular_usage["total_completion_tokens"],
        )

    def get_progress_pct(
        self, phase: str, completed: bool = False, error: bool = False
    ) -> float:
        """Calculates the progress percentage for the debate."""
        phase_weights = {"context": 0.1, "debate": 0.7, "synthesis": 0.2}

        current_progress = 0.0
        if phase == "context":
            current_progress = 0.05
        elif phase == "debate":
            total_debate_personas = (
                len(self.intermediate_steps.get("Persona_Sequence", [])) - 1
            )
            completed_debate_personas = sum(
                1
                for k in self.intermediate_steps
                if k.endswith("_Output")
                and not k.startswith(("Final_", "Context_Analysis_Output"))
                and k
                not in {
                    "Self_Improvement_Metrics",
                    "Debate_History",
                    "Conflict_Resolution_Attempt",
                    "Unresolved_Conflict",
                    "Context_Aware_Assistant_Output",
                }
            )

            if total_debate_personas > 0:
                current_progress = (
                    phase_weights["context"]
                    + (completed_debate_personas / total_debate_personas)
                    * phase_weights["debate"]
                )
            else:
                current_progress = phase_weights["context"]
        elif phase == "synthesis":
            current_progress = phase_weights["context"] + phase_weights["debate"] + 0.1

        if completed:
            current_progress = 1.0
        elif error:
            current_progress = max(current_progress, 0.99)

        return min(max(0.0, current_progress), 1.0)

    def _prepare_llm_call_config(
        self,
        persona_config: PersonaConfig,
        max_output_tokens_for_turn: int,
        requested_model_name: Optional[str],
    ) -> tuple[str, int]:
        """Prepares the model name and effective max output tokens for an LLM call."""
        final_model_to_use = (
            requested_model_name if requested_model_name else self.model_name
        )
        safety_margin_factor = 0.98
        effective_max_output_tokens = int(
            min(
                max_output_tokens_for_turn,
                self.llm_provider.tokenizer.max_output_tokens,
            )
            * safety_margin_factor
        )
        effective_max_output_tokens = max(128, effective_max_output_tokens)
        self._log_with_context(
            "debug",
            f"Adjusting max_output_tokens for {persona_config.name}. Requested: {max_output_tokens_for_turn}, Model Max: {self.llm_provider.tokenizer.max_output_tokens}, Effective: {effective_max_output_tokens}",
        )
        return final_model_to_use, effective_max_output_tokens

    def _prepare_llm_prompt_and_config(
        self,
        persona_name: str,
        prompt_for_llm: str,
        max_output_tokens_for_turn: int,
        requested_model_name: Optional[str],
        context_for_template: Optional[dict[str, Any]] = None,
        # Removed aggressive_optimization: bool = False as it's now internal to PromptOptimizer
    ) -> tuple[PersonaConfig, type[BaseModel], str, str, str, int]:
        """Prepar es persona configuration, output schema, and the final system/user prompts
        for an LLM call.
        """
        persona_config = self.persona_manager.get_adjusted_persona_config(persona_name)
        if not persona_config:
            self._log_with_context(
                "error",
                f"Persona configuration not found for {persona_name}. Cannot execute turn.",
                persona=persona_name,
            )
            raise ChimeraError(f"Persona configuration not found for {persona_name}.")

        # Retrieve the schema name from the persona_config
        schema_name = persona_config.output_schema
        # Use the output_parser to get the actual Pydantic schema class
        output_schema_class = self.output_parser._get_schema_class_from_name(
            schema_name
        )

        # Get the schema class name safely, handling mock objects
        schema_class_name = getattr(
            output_schema_class, "__name__", str(output_schema_class)
        )
        self._log_with_context(
            "info", f"Using schema {schema_class_name} for {persona_name}."
        )

        # Render system prompt from template
        # The get_adjusted_persona_config now returns a PersonaConfig with a _rendered_system_prompt attribute
        final_system_prompt_base = getattr(
            persona_config,
            "_rendered_system_prompt",
            persona_config.system_prompt_template,
        )
        if not final_system_prompt_base:
            self.logger.error(f"Rendered system prompt is empty for {persona_name}.")
            final_system_prompt_base = "You are a helpful AI assistant."  # Fallback

        full_system_prompt_parts = [final_system_prompt_base]
        full_system_prompt_parts.append(SHARED_JSON_INSTRUCTIONS)
        full_system_prompt_parts.append(
            f"**JSON Schema for {schema_class_name}:**\n```json\n{json.dumps(output_schema_class.model_json_schema(), indent=2)}\n```"
        )
        final_system_prompt = "\n\n".join(full_system_prompt_parts)

        # Optimize the user prompt before sending
        # MODIFIED: Pass persona_config directly to optimize_prompt
        optimized_user_prompt = self.prompt_optimizer.optimize_prompt(
            user_prompt_text=prompt_for_llm,
            persona_config=persona_config,
            max_output_tokens_for_turn=max_output_tokens_for_turn,
            system_message_for_token_count=final_system_prompt,
            is_self_analysis_prompt=self.is_self_analysis,
            # Removed aggressive_optimization as it's now internal to PromptOptimizer
        )

        final_model_to_use, effective_max_output_tokens = self._prepare_llm_call_config(
            persona_config, max_output_tokens_for_turn, requested_model_name
        )
        return (
            persona_config,
            output_schema_class,
            final_system_prompt,
            optimized_user_prompt,
            final_model_to_use,
            effective_max_output_tokens,
        )

    # REMOVED: _call_llm_and_track_tokens as it's now handled by call_llm in src/llm/client.py

    def _process_llm_response(
        self,
        persona_name: str,
        raw_llm_output: str,
        output_schema_class: type[BaseModel],
        is_truncated: bool,
        tokens_used_in_turn: int,
        is_successful_turn: bool,  # NEW: Added is_successful_turn for persona performance tracking
    ) -> dict[str, Any]:
        """Parses LLM output, tracks malformed blocks, and performs content alignment checks."""
        parsed_output = self.output_parser.parse_and_validate(
            raw_llm_output, output_schema_class
        )
        has_schema_error = bool(parsed_output.get("malformed_blocks"))

        if has_schema_error:
            self.intermediate_steps.setdefault("malformed_blocks", []).extend(
                parsed_output["malformed_blocks"]
            )
            self._log_with_context(
                "warning",
                f"Parser reported malformed blocks for {persona_name}.",
                persona=persona_name,
                malformed_blocks=parsed_output["malformed_blocks"],
            )

        is_aligned, validation_message, nuanced_feedback = (
            self.content_validator.validate(persona_name, parsed_output)
        )
        self.persona_manager.record_persona_performance(
            persona_name,
            1,  # Turn number is not directly passed here, assuming 1 for simplicity
            parsed_output,
            is_aligned
            and not has_schema_error
            and is_successful_turn,  # Use is_successful_turn
            validation_message,
            is_truncated=is_truncated,
            schema_validation_failed=has_schema_error,
            token_budget_exceeded=False,
            tokens_used_in_turn=tokens_used_in_turn,
        )

        if not is_aligned:
            self._log_with_context(
                "warning",
                f"Content misalignment detected for {persona_name}: {validation_message}",
                persona=persona_name,
                validation_message=validation_message,
            )
            self.intermediate_steps.setdefault("malformed_blocks", []).extend(
                [
                    {
                        "type": "CONTENT_MISALIGNMENT",
                        "message": f"Output from {persona_name} drifted from the core topic: {validation_message}",
                        "persona": persona_name,
                        "raw_string_snippet": str(parsed_output)[:500],
                    }
                ]
                + nuanced_feedback.get("malformed_blocks", [])
            )
            if isinstance(parsed_output, dict):
                parsed_output["content_misalignment_warning"] = validation_message
            else:
                parsed_output = {
                    "general_output": f"WARNING: Content misalignment detected: {validation_message}\n\n{parsed_output}",
                    "malformed_blocks": parsed_output.get("malformed_blocks", []),
                }
        return parsed_output

    def _generate_retry_feedback(
        self, error: SchemaValidationError, original_prompt: str
    ) -> str:
        """Generates a new prompt for retrying an LLM turn, including feedback about the validation error."""
        error_details_str = json.dumps(error.to_dict().get("details", {}), indent=2)
        feedback_message = (
            f"Your previous response failed schema validation. "
            f"The error was: {error.message}\n"
            f"Details: {error_details_str}\n\n"
            f"Please correct your response to strictly adhere to the required JSON schema. "
            f"DO NOT include any conversational text or markdown fences outside the JSON object. "
            f"Focus solely on providing a correct, valid JSON response. Original request: {original_prompt}"
        )
        self._log_with_context(
            "debug",
            f"Generated retry feedback for {error.error_code}.",
            feedback_message_snippet=feedback_message[:200],
        )
        return feedback_message

    def _execute_llm_turn(
        self,
        persona_name: str,
        prompt_for_llm: str,
        phase: str,
        max_output_tokens_for_turn: int,
        requested_model_name: Optional[str] = None,
        max_retries: int = 2,  # This max_retries is now for internal logic, not LLM API retries
        context_for_template: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Executes a single LLM turn for a given persona, handles API calls,
        parsing, validation, and token tracking, with retry logic for validation failures.
        """
        turn_start_time = datetime.now()
        self._log_with_context(
            "debug",
            f"Executing LLM turn for {persona_name} in {phase} phase.",
            persona=persona_name,
            phase=phase,
        )

        raw_llm_output = ""
        is_truncated = False
        current_prompt_for_retry = prompt_for_llm
        tokens_used_in_turn = 0
        # is_successful_turn = False # Track if the turn was ultimately successful (F841 fix: removed unused variable)
        turn_success = False

        # Aggressive token optimization logic is now handled internally by PromptOptimizer
        # No need for aggressive_optimization_flag here.

        # The retry loop for schema validation is still here, but it calls the centralized `call_llm`
        for attempt in range(max_retries + 1):
            try:
                if self.status_callback:
                    self.status_callback(
                        f"LLM Call: [bold]{persona_name.replace('_', ' ')}[/bold] generating response (Attempt {attempt + 1}/{max_retries + 1})...",
                        state="running",
                        current_total_tokens=self.token_tracker.current_usage,
                        current_total_cost=self.get_total_estimated_cost(),
                        progress_pct=self.get_progress_pct(phase),
                        current_persona_name=persona_name,
                    )

                (
                    persona_config,
                    output_schema_class,
                    final_system_prompt,
                    optimized_user_prompt,
                    final_model_to_use,
                    effective_max_output_tokens,
                ) = self._prepare_llm_prompt_and_config(
                    persona_name,
                    current_prompt_for_retry,
                    max_output_tokens_for_turn,
                    requested_model_name,
                    context_for_template=context_for_template,
                )

                # NEW: Call the centralized LLM orchestration function
                llm_response = self.llm_orchestrator.call_llm(
                    template_name="generic_llm_call",  # A generic template name, as prompt is already optimized
                    prompt_params={"prompt": optimized_user_prompt},
                    system_prompt=final_system_prompt,
                    output_schema=output_schema_class,
                    temperature=persona_config.temperature,
                    max_tokens=effective_max_output_tokens,
                    persona_config=persona_config,
                    requested_model_name=final_model_to_use,
                    context=persona_name,  # Pass persona name as context for token tracking
                )

                raw_llm_output = llm_response["text"]
                input_tokens = llm_response["usage"]["prompt_tokens"]
                output_tokens = llm_response["usage"]["completion_tokens"]
                tokens_used_in_turn = llm_response["usage"]["total_tokens"]
                is_truncated = llm_response["is_truncated"]

                # NEW: Centralized validation right after the LLM call (already done by call_llm, but re-validate for safety)
                validated_data = validate_llm_output(
                    raw_llm_output, output_schema_class
                )
                if not validated_data:
                    raise SchemaValidationError(
                        error_type="POST_CALL_VALIDATION_FAILED",
                        field_path="LLM_OUTPUT",
                        invalid_value=raw_llm_output[:500],
                    )

                parsed_output = self._process_llm_response(
                    persona_name,
                    raw_llm_output,
                    output_schema_class,
                    is_truncated,
                    tokens_used_in_turn,
                    is_successful_turn=True,  # Assume successful until content validation
                )

                has_schema_error_after_processing = bool(
                    parsed_output.get("malformed_blocks")
                )

                if not has_schema_error_after_processing:
                    # is_successful_turn = True # F841 fix: removed unused assignment
                    turn_success = True
                    break  # Exit retry loop on success

                if attempt < max_retries:
                    self._log_with_context(
                        "warning",
                        f"Validation error for {persona_name} (Attempt {attempt + 1}/{max_retries + 1}). Retrying. Output had malformed blocks. ",
                        persona=persona_name,
                        phase=phase,
                        malformed_blocks=parsed_output.get("malformed_blocks"),
                    )
                    simulated_error = SchemaValidationError(
                        error_type="LLM_OUTPUT_MALFORMED",
                        field_path="LLM_OUTPUT",
                        invalid_value=raw_llm_output[:500],
                        details={
                            "malformed_blocks": parsed_output.get("malformed_blocks")
                        },
                    )
                    current_prompt_for_retry = self._generate_retry_feedback(
                        simulated_error, current_prompt_for_retry
                    )
                    self.intermediate_steps.setdefault("malformed_blocks", []).append(
                        {
                            "type": "RETRYABLE_VALIDATION_ERROR",
                            "message": "LLM output had malformed blocks, retrying.",
                            "attempt": attempt + 1,
                            "persona": persona_name,
                            "original_malformed_blocks": parsed_output.get(
                                "malformed_blocks"
                            ),
                        }
                    )
                    # Record performance for failed attempt
                    self.persona_manager.record_persona_performance(
                        persona_name,
                        attempt + 1,
                        raw_llm_output,
                        False,  # Not aligned
                        "Schema validation failed (malformed blocks)",
                        is_truncated=is_truncated,
                        schema_validation_failed=True,
                        token_budget_exceeded=False,
                        tokens_used_in_turn=tokens_used_in_turn,
                        is_successful_turn=False,
                    )
                    continue
                else:
                    self._log_with_context(
                        "error",
                        f"Max retries ({max_retries}) reached for {persona_name}. Returning fallback JSON.",
                        persona=persona_name,
                    )
                    # Record performance for final failed attempt
                    self.persona_manager.record_persona_performance(
                        persona_name,
                        attempt + 1,
                        raw_llm_output,
                        False,  # Not aligned
                        "Schema validation failed after multiple attempts",
                        is_truncated=is_truncated,
                        schema_validation_failed=True,
                        token_budget_exceeded=False,
                        tokens_used_in_turn=tokens_used_in_turn,
                        is_successful_turn=False,
                    )
                    return self.output_parser._create_fallback_output(
                        output_schema_class,
                        malformed_blocks=[
                            {
                                "type": "MAX_RETRIES_REACHED",
                                "message": f"Schema validation failed after {max_retries} retries.",
                            }
                        ],
                        raw_output_snippet=raw_llm_output,
                        partial_data=None,
                        extracted_json_str=None,
                    )

            except (
                LLMProviderRequestError,
                LLMProviderResponseError,
                CircuitBreakerError,
                ChimeraError,
                SchemaValidationError,  # Catch SchemaValidationError directly from call_llm
            ) as e:
                self._log_with_context(
                    "error",
                    f"Non-retryable error during LLM turn for {persona_name}: {e}",
                    persona=persona_name,
                    phase=phase,
                    exc_info=True,
                    original_exception=e,
                )
                self.persona_manager.record_persona_performance(
                    persona_name,
                    attempt + 1,
                    raw_llm_output,
                    False,
                    f"Non-retryable error: {type(e).__name__}",
                    is_truncated=is_truncated,
                    schema_validation_failed=isinstance(e, SchemaValidationError),
                    token_budget_exceeded=False,
                    tokens_used_in_turn=tokens_used_in_turn,
                    is_successful_turn=False,
                )
                self.system_monitor.record_error(
                    type(e).__name__, str(e), self.request_id
                )
                raise e

            except TokenBudgetExceededError as e:
                self._log_with_context(
                    "error",
                    f"Token budget exceeded during LLM turn for {persona_name}: {e}",
                    persona=persona_name,
                    phase=phase,
                    exc_info=True,
                    original_exception=e,
                )
                self.persona_manager.record_persona_performance(
                    persona_name,
                    attempt + 1,
                    raw_llm_output,
                    False,
                    f"Token budget exceeded: {str(e)}",
                    is_truncated=is_truncated,
                    schema_validation_failed=False,
                    token_budget_exceeded=True,
                    tokens_used_in_turn=tokens_used_in_turn,
                    is_successful_turn=False,
                )
                self.system_monitor.record_error(
                    "TokenBudgetExceeded", str(e), self.request_id
                )
                raise e

            except Exception as e:
                self._log_with_context(
                    "error",
                    f"An unexpected error occurred during LLM turn for {persona_name}: {e}",
                    persona=persona_name,
                    phase=phase,
                    exc_info=True,
                    original_exception=e,
                )
                self.persona_manager.record_persona_performance(
                    persona_name,
                    attempt + 1,
                    raw_llm_output,
                    False,
                    f"Unexpected error: {type(e).__name__}",
                    is_truncated=is_truncated,
                    schema_validation_failed=False,
                    token_budget_exceeded=False,
                    tokens_used_in_turn=tokens_used_in_turn,
                    is_successful_turn=False,
                )
                self.system_monitor.record_error(
                    type(e).__name__, str(e), self.request_id
                )
                raise ChimeraError(
                    f"Unexpected error in LLM turn for {persona_name}: {e}",
                    original_exception=e,
                ) from e

        self.intermediate_steps[f"{persona_name}_Output"] = parsed_output
        self.intermediate_steps[f"{persona_name}_Tokens_Used"] = tokens_used_in_turn
        self.intermediate_steps[f"{persona_name}_Estimated_Cost_USD"] = (
            self.llm_provider.calculate_usd_cost(input_tokens, output_tokens)
        )

        # Calculate turn duration
        turn_duration = (datetime.now() - turn_start_time).total_seconds()

        # Record persona performance metrics
        self.system_monitor.record_persona_performance(
            persona_name,
            turn_success,
            tokens_used_in_turn,
            turn_duration,
            self.request_id,
        )

        self._log_with_context(
            "info",
            f"LLM turn for {persona_name} completed successfully.",
            persona=persona_name,
            phase=phase,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens_for_turn=tokens_used_in_turn,
            turn_duration=turn_duration,
        )

        # NEW: Log performance metrics using enhanced logger
        perf_logger = get_performance_logger()
        perf_logger.log_persona_performance(
            persona_name,
            turn_success,
            turn_duration,
            tokens_used_in_turn,
            self.request_id,
        )

        return parsed_output

    def _initialize_debate_state(self):
        """Initializes or resets the debate's internal state variables."""
        self.intermediate_steps = {}
        self.token_tracker.reset()
        self.rich_console.print(
            f"[bold green]Starting Socratic Debate for prompt:[/bold green] [italic]{self.initial_prompt}[/italic]"
        )
        self._log_with_context("info", "Debate state initialized.")

    def _perform_context_analysis_phase(
        self, persona_sequence: tuple[str, ...]
    ) -> Optional[dict[str, Any]]:
        """Performs context analysis based on the initial prompt and codebase context."""
        if not self.raw_file_contents or not self.context_analyzer:
            self._log_with_context(
                "info",
                "No codebase context or analyzer available. Skipping context analysis.",
            )
            return None

        if self.raw_file_contents and self.context_analyzer:
            current_files_hash = hash(frozenset(self.raw_file_contents.items()))
            if (
                not hasattr(self.context_analyzer, "_last_raw_file_contents_hash")
                or self.context_analyzer._last_raw_file_contents_hash
                != current_files_hash
            ):
                try:
                    self.context_analyzer.compute_file_embeddings(
                        self.raw_file_contents
                    )
                    self.context_analyzer._last_raw_file_contents_hash = (
                        current_files_hash
                    )
                    self._log_with_context(
                        "info",
                        "Computed file embeddings for codebase context during context analysis phase.",
                    )
                except Exception as e:
                    self._log_with_context(
                        "error",
                        f"Error during context analysis: {e}[/red]",
                        exc_info=True,
                        original_exception=e,
                    )
                    if self.status_callback:
                        self.status_callback(
                            f"[red]Error computing context embeddings: {e}[/red]",
                            state="error",
                            current_total_tokens=self.token_tracker.current_usage,
                            current_total_cost=self.get_total_estimated_cost(),
                            progress_pct=self.get_progress_pct("context", error=True),
                            current_persona_name="Context_Relevance_Analyzer_Phase",
                        )
                    return {"error": f"Context analysis failed: {e}"}

        self._log_with_context("info", "Performing context analysis.")
        try:
            relevant_files = self.context_analyzer.find_relevant_files(
                self.initial_prompt,
                max_context_tokens=self.phase_budgets["context"],
                active_personas=persona_sequence,
            )

            context_summary_str = self.context_analyzer.generate_context_summary(
                relevant_files, self.phase_budgets["context"], self.initial_prompt
            )

            estimated_tokens = self.tokenizer.count_tokens(context_summary_str)

            self.check_budget("context", estimated_tokens, "Context Analysis Summary")
            self.track_token_usage("context", estimated_tokens)

            # NEW: Log token usage for 80/20 analysis
            perf_logger = get_performance_logger()
            perf_logger.log_token_usage(
                estimated_tokens, self.max_total_tokens_budget, self.request_id
            )

            context_analysis_output = {
                "relevant_files": relevant_files,
                "context_summary": context_summary_str,
                "estimated_tokens": estimated_tokens,
            }
            self._log_with_context(
                "info",
                "Context analysis completed.",
                relevant_files=[f[0] for f in relevant_files],
                estimated_tokens=estimated_tokens,
            )
            return context_analysis_output
        except Exception as e:
            self._log_with_context(
                "error",
                f"Error during context analysis: {e}",
                exc_info=True,
                original_exception=e,
            )
            self.rich_console.print(f"[red]Error during context analysis: {e}[/red]")
            return {"error": f"Context analysis failed: {e}"}

    def _get_final_persona_sequence(
        self, prompt: str, context_analysis_results: Optional[dict[str, Any]]
    ) -> list[str]:
        """Delegates to the PersonaRouter to determine the optimal persona sequence,
        incorporating prompt analysis, domain, and context analysis results.
        """
        if not self.persona_router:
            self._log_with_context(
                "error",
                "PersonaRouter not initialized. Falling back to default sequence.",
            )
            return [
                "Visionary_Generator",
                "Skeptical_Generator",
                "Impartial_Arbitrator",
            ]

        sequence = self.persona_router.determine_persona_sequence(
            prompt=prompt,
            domain=self.domain,
            intermediate_results=self.intermediate_steps,
            context_analysis_results=context_analysis_results,
        )
        return sequence

    def _distribute_debate_persona_budgets(self, persona_sequence: list[str]):
        """Distributes the total debate token budget among the actual personas in the sequence.
        This is called *after* the final persona sequence is determined.
        """
        MIN_PERSONA_TOKENS = 256

        active_debate_personas = [
            p
            for p in persona_sequence
            if p
            not in [
                "Context_Aware_Assistant",
                "Impartial_Arbitrator",
                "General_Synthesizer",
                "Self_Improvement_Analyst",
            ]
        ]

        num_debate_personas = len(active_debate_personas)
        if num_debate_personas == 0:
            self.phase_budgets["persona_turn_budgets"] = {}
            self.phase_budgets["debate"] = 0
            return

        base_allocation_per_persona = (
            self.phase_budgets["debate"] // num_debate_personas
        )
        remaining_budget = self.phase_budgets["debate"]

        persona_turn_budgets: dict[str, int] = {}

        for p_name in active_debate_personas:
            persona_config = self.persona_manager.get_adjusted_persona_config(p_name)
            allocated = max(
                MIN_PERSONA_TOKENS,
                min(base_allocation_per_persona, persona_config.max_tokens),
            )
            persona_turn_budgets[p_name] = allocated
            remaining_budget -= allocated

        if remaining_budget > 0:
            redistribution_pool_personas = [
                p_name
                for p_name in active_debate_personas
                if persona_turn_budgets[p_name]
                < self.persona_manager.get_adjusted_persona_config(p_name).max_tokens
            ]
            if redistribution_pool_personas:
                share_per_redistribution_persona = remaining_budget // len(
                    redistribution_pool_personas
                )
                for p_name in redistribution_pool_personas:
                    persona_config = self.persona_manager.get_adjusted_persona_config(
                        p_name
                    )
                    persona_turn_budgets[p_name] = min(
                        persona_config.max_tokens,
                        persona_turn_budgets[p_name] + share_per_redistribution_persona,
                    )

        current_total_persona_budget = sum(persona_turn_budgets.values())
        if current_total_persona_budget > self.phase_budgets["debate"]:
            reduction_factor = 1.0
            if current_total_persona_budget > 0:
                reduction_factor = (
                    self.phase_budgets["debate"] / current_total_persona_budget
                )
            for p_name in active_debate_personas:
                persona_turn_budgets[p_name] = max(
                    MIN_PERSONA_TOKENS,
                    int(persona_turn_budgets[p_name] * reduction_factor),
                )

        self.phase_budgets["persona_turn_budgets"] = persona_turn_budgets
        self._log_with_context(
            "info",
            "Debate persona turn budgets distributed.",
            persona_turn_budgets=self.phase_budgets["persona_turn_budgets"],
        )

    def _process_context_persona_turn(
        self,
        persona_sequence: list[str],
        context_analysis_results: Optional[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        """Executes the Context_Aware_Assistant persona turn if it's in the sequence."""
        if "Context_Aware_Assistant" not in persona_sequence:
            self._log_with_context(
                "info", "Context_Aware_Assistant not in sequence. Skipping turn."
            )
            return None

        self._log_with_context("info", "Executing Context_Aware_Assistant turn.")
        persona_config = self.persona_manager.get_adjusted_persona_config(
            "Context_Aware_Assistant"
        )
        if not persona_config:
            self._log_with_context(
                "error", "Context_Aware_Assistant persona configuration not found."
            )
            return {"error": "Context_Aware_Assistant config missing."}

        context_prompt_content = ""
        if context_analysis_results and context_analysis_results.get("relevant_files"):
            for file_path, _ in context_analysis_results["relevant_files"]:
                content = self.raw_file_contents.get(file_path, "")
                if content:
                    context_prompt_content += (
                        f"### File: {file_path}\n```\n{content}\n```\n\n"
                    )

        prompt = f"Analyze the following codebase context and provide a structured analysis:\n\n{context_prompt_content}"

        max_output_tokens_for_turn = self.phase_budgets.get(
            "persona_turn_budgets", {}
        ).get("Context_Aware_Assistant", persona_config.max_tokens)
        estimated_tokens = (
            self.tokenizer.count_tokens(prompt) + max_output_tokens_for_turn
        )
        self.check_budget("debate", estimated_tokens, "Context_Aware_Assistant")

        # NEW: Log token usage for 80/20 analysis
        perf_logger = get_performance_logger()
        perf_logger.log_token_usage(
            estimated_tokens, self.max_total_tokens_budget, self.request_id
        )

        try:
            output = self._execute_llm_turn(
                "Context_Aware_Assistant", prompt, "debate", max_output_tokens_for_turn
            )
            self._log_with_context("info", "Context_Aware_Assistant turn completed.")
            return output
        except Exception as e:
            self._log_with_context(
                "error",
                f"Error during Context_Aware_Assistant turn: {e}",
                exc_info=True,
                original_exception=e,
            )
            self.rich_console.print(
                f"[red]Error during Context_Aware_Assistant turn: {e}[/red]"
            )
            return {"error": f"Context_Aware_Assistant turn failed: {e}"}

    def _build_persona_context_string(
        self, persona_name: str, context_persona_turn_results: dict[str, Any]
    ) -> str:
        """Builds a persona-specific context string from context analysis results."""
        persona_specific_context_str = ""
        if persona_name == "Security_Auditor" and context_persona_turn_results.get(
            "security_summary"
        ):
            persona_specific_context_str = f"Security Context Summary:\n{safe_json_dumps(context_persona_turn_results['security_summary'], indent=2, default=convert_to_json_friendly)}"
        elif persona_name == "Code_Architect" and context_persona_turn_results.get(
            "architecture_summary"
        ):
            persona_specific_context_str = f"Architecture Context Summary:\n{safe_json_dumps(context_persona_turn_results['architecture_summary'], indent=2, default=convert_to_json_friendly)}"
        elif persona_name == "DevOps_Engineer" and context_persona_turn_results.get(
            "devops_summary"
        ):
            persona_specific_context_str = f"DevOps Context Summary:\n{safe_json_dumps(context_persona_turn_results['devops_summary'], indent=2, default=convert_to_json_friendly)}"
        elif persona_name == "Test_Engineer" and context_persona_turn_results.get(
            "testing_summary"
        ):
            persona_specific_context_str = f"Testing Context Summary:\n{safe_json_dumps(context_persona_turn_results['testing_summary'], indent=2, default=convert_to_json_friendly)}"
        elif context_persona_turn_results.get("general_overview"):
            persona_specific_context_str = f"General Codebase Overview:\n{context_persona_turn_results['general_overview']}"
        if context_persona_turn_results.get("configuration_summary"):
            persona_specific_context_str += f"\n\nStructured Configuration Analysis:\n{safe_json_dumps(context_persona_turn_results['configuration_summary'], indent=2, default=convert_to_json_friendly)}"
        if context_persona_turn_results.get("deployment_summary"):
            persona_specific_context_str += f"\n\nStructured Deployment Robustness Analysis:\n{safe_json_dumps(context_persona_turn_results['deployment_summary'], indent=2, default=convert_to_json_friendly)}"
        return persona_specific_context_str

    def _summarize_previous_output(
        self, previous_output_for_llm: Union[str, dict[str, Any]], is_problematic: bool
    ) -> str:
        """Summarizes the previous LLM output for the current prompt."""
        SUMMARY_TOKEN_LIMIT = 100

        if is_problematic:
            summary = "Previous persona's output had issues (malformed JSON or content misalignment). "
            if isinstance(previous_output_for_llm, dict):
                if previous_output_for_llm.get("CRITIQUE_SUMMARY"):
                    truncated_summary = self.tokenizer.truncate_to_token_limit(
                        previous_output_for_llm["CRITIQUE_SUMMARY"], SUMMARY_TOKEN_LIMIT
                    )
                    summary += f"Summary of previous critique: {truncated_summary}"
                elif previous_output_for_llm.get("ANALYSIS_SUMMARY"):
                    truncated_summary = self.tokenizer.truncate_to_token_limit(
                        previous_output_for_llm["ANALYSIS_SUMMARY"], SUMMARY_TOKEN_LIMIT
                    )
                    summary += f"Summary of previous analysis: {truncated_summary}"
                elif previous_output_for_llm.get("general_output"):
                    truncated_summary = self.tokenizer.truncate_to_token_limit(
                        previous_output_for_llm["general_output"], SUMMARY_TOKEN_LIMIT
                    )
                    summary += (
                        f"Summary of previous general output: {truncated_summary}"
                    )
                elif previous_output_for_llm.get("summary"):
                    truncated_summary = self.tokenizer.truncate_to_token_limit(
                        previous_output_for_llm["summary"], SUMMARY_TOKEN_LIMIT
                    )
                    summary += (
                        f"Summary of previous conflict report: {truncated_summary}"
                    )
                else:
                    summary += "Details in malformed_blocks."
            else:
                truncated_snippet = self.tokenizer.truncate_to_token_limit(
                    str(previous_output_for_llm), SUMMARY_TOKEN_LIMIT * 2
                )
                summary += f"Raw error snippet: {truncated_snippet}..."
            return f"Previous Debate Output Summary (with issues):\n{summary}\n\n"
        else:
            output_content = previous_output_for_llm

            if isinstance(output_content, dict):
                output_copy = output_content.copy()
                for key in [
                    "CRITIQUE_SUMMARY",
                    "ANALYSIS_SUMMARY",
                    "general_output",
                    "summary",
                ]:
                    if output_copy.get(key):
                        output_copy[key] = self.tokenizer.truncate_to_token_limit(
                            output_copy[key], SUMMARY_TOKEN_LIMIT
                        )

                for key_list in ["SUGGESTIONS", "IMPACTFUL_SUGGESTIONS"]:
                    if key_list in output_copy and isinstance(
                        output_copy[key_list], list
                    ):
                        for suggestion in output_copy[key_list]:
                            if isinstance(suggestion, dict):
                                for field in [
                                    "PROBLEM",
                                    "PROPOSED_SOLUTION",
                                    "EXPECTED_IMPACT",
                                    "RATIONALE",
                                ]:
                                    if suggestion.get(field):
                                        suggestion[field] = (
                                            self.tokenizer.truncate_to_token_limit(
                                                suggestion[field], SUMMARY_TOKEN_LIMIT
                                            )
                                        )
                                if (
                                    "CODE_CHANGES_SUGGESTED" in suggestion
                                    and isinstance(
                                        suggestion["CODE_CHANGES_SUGGESTED"], list
                                    )
                                ):
                                    for change in suggestion["CODE_CHANGES_SUGGESTED"]:
                                        if isinstance(change, dict):
                                            if change.get("FULL_CONTENT"):
                                                change["FULL_CONTENT"] = (
                                                    self.tokenizer.truncate_to_token_limit(
                                                        change["FULL_CONTENT"],
                                                        SUMMARY_TOKEN_LIMIT,
                                                    )
                                                )
                                            if change.get("DIFF_CONTENT"):
                                                change["DIFF_CONTENT"] = (
                                                    self.tokenizer.truncate_to_token_limit(
                                                        change["DIFF_CONTENT"],
                                                        SUMMARY_TOKEN_LIMIT,
                                                    )
                                                )

                return f"Previous Debate Output:\n{safe_json_dumps(output_copy, indent=2, default=convert_to_json_friendly)}\n\n"
            else:
                truncated_output = self.tokenizer.truncate_to_token_limit(
                    str(previous_output_for_llm), SUMMARY_TOKEN_LIMIT * 2
                )
                return f"Previous Debate Output:\n{truncated_output}\n\n"

    def _handle_devils_advocate_turn(
        self, output: dict[str, Any], debate_history_so_far: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Handles the specific logic for the Devils_Advocate persona's output.
        If a conflict is found, it returns the ConflictReport for the main loop to handle resolution.
        """
        from src.models import ConflictReport

        try:
            conflict_report = ConflictReport.model_validate(output)
            if conflict_report.conflict_found:
                self._log_with_context(
                    "info",
                    f"Devils_Advocate reported conflict: {conflict_report.summary}. Main loop will trigger ConflictResolutionManager.",
                    conflict_report=conflict_report.model_dump(),
                )
                return output
            else:
                self._log_with_context(
                    "info",
                    f"Devils_Advocate reported no conflict: {conflict_report.summary}",
                )
                self.intermediate_steps["Unresolved_Conflict"] = None
                self.intermediate_steps["Conflict_Resolution_Attempt"] = None
                return {
                    "status": "no_conflict_reported",
                    "summary": conflict_report.summary,
                }
        except ValidationError:
            self._log_with_context(
                "warning",
                "Devils_Advocate output was not a valid ConflictReport. Treating as general output.",
            )
            return output
        except Exception as e:
            self._log_with_context(
                "error", f"Error processing Devils_Advocate output: {e}", exc_info=True
            )
            return {"error": f"Error processing Devils_Advocate output: {e}"}

    def _prepare_prompt_for_turn(
        self,
        persona_name: str,
        previous_output: Any,
        context_results: Optional[dict[str, Any]],
        user_query: str,  # Add user_query to pass to RAG
    ) -> str:
        """Prepares the full prompt for a single persona turn, incorporating RAG context."""
        rag_context = ""
        if hasattr(self, "rag_orchestrator") and self.rag_orchestrator:
            rag_context = self.rag_orchestrator.generate_rag_context(
                user_query, persona_name
            )

        persona_specific_context = self._build_persona_context_string(
            persona_name, context_results
        )
        is_problematic = isinstance(previous_output, dict) and (
            previous_output.get("malformed_blocks")
            or previous_output.get("content_misalignment_warning")
            or (
                "conflict_found" in previous_output
                and previous_output["conflict_found"] is True
            )
        )
        previous_output_summary = self._summarize_previous_output(
            previous_output, is_problematic
        )

        prompt_parts = [f"Initial Problem: {self.initial_prompt}\n"]
        if persona_specific_context:
            prompt_parts.append(f"Relevant Code Context:\n{persona_specific_context}\n")
        if rag_context:
            prompt_parts.append(rag_context)  # Add RAG context here
        prompt_parts.append(previous_output_summary)

        # Inject the valid file list into the prompt for code-generating personas
        # This list is derived from the actual scanned codebase.
        if persona_name in [
            "Code_Architect",
            "Security_Auditor",
            "DevOps_Engineer",
            "Test_Engineer",
            "Self_Improvement_Analyst",
        ]:
            valid_files_list = list(self.raw_file_contents.keys())
            file_list_context = (
                "CRITICAL INSTRUCTION: You MUST only reference file paths from the following list of valid files in the codebase. "
                f"Do not invent or assume paths.\nVALID FILES: {json.dumps(valid_files_list, indent=2)}\n\n"
            )
            prompt_parts.insert(
                0, file_list_context
            )  # Prepend this critical instruction
        return "\n".join(prompt_parts)

    def _handle_problematic_output(
        self,
        current_problematic_output: dict[str, Any],
        debate_history_so_far: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Handles conflict resolution for a problematic turn output."""
        # Pass the current problematic output along with the history up to this point
        resolved_output_from_manager = self.conflict_manager.resolve_conflict(
            debate_history_so_far
            + [
                {
                    "persona": "Current_Problematic_Persona",
                    "output": current_problematic_output,
                }
            ]
        )
        if resolved_output_from_manager and resolved_output_from_manager.get(
            "resolved_output"
        ):
            # The conflict manager will append its own turn to the history in _execute_debate_persona_turns
            self.intermediate_steps["Conflict_Resolution_Attempt"] = {
                "conflict_resolved": True,
                **resolved_output_from_manager,
            }
            return resolved_output_from_manager["resolved_output"]
        else:
            self.intermediate_steps["Unresolved_Conflict"] = current_problematic_output
            return current_problematic_output

    def _execute_debate_persona_turns(
        self,
        persona_sequence: list[str],
        context_persona_turn_results: Optional[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Executes the main debate turns for each persona in the sequence."""
        debate_history = []

        previous_output_for_llm: Union[str, dict[str, Any]]
        if context_persona_turn_results:
            previous_output_for_llm = f"Initial Prompt: {self.initial_prompt}\n\nStructured Context Analysis:\n{safe_json_dumps(context_persona_turn_results, indent=2, default=convert_to_json_friendly)}"
        else:
            previous_output_for_llm = f"Initial Prompt: {self.initial_prompt}"

        personas_for_debate = [
            p
            for p in persona_sequence
            if p
            not in [
                "Context_Aware_Assistant",
                "Impartial_Arbitrator",
                "General_Synthesizer",
                "Self_Improvement_Analyst",
            ]
        ]

        if (
            "Devils_Advocate" in persona_sequence
            and "Devils_Advocate" not in personas_for_debate
        ):
            insert_idx = len(personas_for_debate)
            for _i, p_name in reversed(
                list(enumerate(personas_for_debate))
            ):  # FIX B007
                if (
                    "Critic" in p_name
                    or "Analyst" in p_name
                    or "Engineer" in p_name
                    or "Architect" in p_name
                ):
                    insert_idx = _i + 1
                    break
            personas_for_debate.insert(insert_idx, "Devils_Advocate")

        for _i, persona_name in enumerate(personas_for_debate):  # FIX B007
            self._log_with_context(
                "info",
                f"Executing debate turn for persona: {persona_name}",
                persona=persona_name,
            )
            self.status_callback(
                f"Executing: [bold]{persona_name.replace('_', ' ')}[/bold]...",
                state="running",
                current_total_tokens=self.token_tracker.current_usage,
                current_total_cost=self.get_total_estimated_cost(),
                progress_pct=self.get_progress_pct("debate"),
                current_persona_name=persona_name,
            )

            # Refactored prompt preparation
            current_prompt = self._prepare_prompt_for_turn(
                persona_name,
                previous_output_for_llm,
                # Pass the full raw_file_contents for detailed context, not just summary
                # The persona's prompt template can then select what it needs.
                # This ensures maximum available context is provided.
                self.raw_file_contents,  # Pass full raw file contents
                self.initial_prompt,  # Pass initial_prompt for RAG
            )

            base_persona_config = self.persona_manager.all_personas.get(
                persona_name.replace("_TRUNCATED", "")
            )
            max_output_tokens_for_turn = self.phase_budgets.get(
                "persona_turn_budgets", {}
            ).get(
                persona_name,
                base_persona_config.max_tokens if base_persona_config else 1024,
            )

            estimated_tokens = (
                self.tokenizer.count_tokens(current_prompt) + max_output_tokens_for_turn
            )
            self.check_budget("debate", estimated_tokens, persona_name)

            turn_output = None
            turn_status = "success"  # Default status
            raw_llm_output_for_exception = ""  # Initialize for exception handling
            # Removed F841 variables: tokens_used_in_turn, is_truncated_for_exception

            try:
                turn_output = self._execute_llm_turn(
                    persona_name, current_prompt, "debate", max_output_tokens_for_turn
                )
                # Capture values for potential exception handling
                raw_llm_output_for_exception = turn_output  # Assuming _execute_llm_turn returns raw_llm_output as first element

                if persona_name == "Devils_Advocate" and isinstance(turn_output, dict):
                    turn_output = self._handle_devils_advocate_turn(
                        turn_output,
                        debate_history,  # debate_history here is without the current turn
                    )

                # Check if the output is problematic (schema error, misalignment, or conflict_found=True)
                if self._is_problematic_output(turn_output):
                    turn_status = "problematic"
                    # Attempt to resolve the problematic output.
                    # Pass the current problematic turn's data along with the history up to this point.
                    resolved_output_from_manager = (
                        self.conflict_manager.resolve_conflict(
                            debate_history
                            + [
                                {
                                    "persona": persona_name,
                                    "output": turn_output,
                                    "status": "problematic",
                                }
                            ]
                        )
                    )

                    if (
                        resolved_output_from_manager
                        and resolved_output_from_manager.get("resolved_output")
                    ):
                        # If conflict manager successfully resolved, update turn_output and status
                        turn_output = resolved_output_from_manager["resolved_output"]
                        turn_status = "corrected"
                        # Append the conflict resolution manager's turn to history
                        debate_history.append(
                            {
                                "persona": "Conflict_Resolution_Manager",
                                "output": resolved_output_from_manager,
                                "status": "success",  # Conflict manager's output is considered successful
                            }
                        )
                        self.intermediate_steps["Conflict_Resolution_Attempt"] = {
                            "conflict_resolved": True,
                            **resolved_output_from_manager,
                        }
                        self.intermediate_steps["Unresolved_Conflict"] = None
                    else:
                        # If conflict manager failed to resolve, keep status as problematic
                        self.intermediate_steps["Unresolved_Conflict"] = turn_output
                        self.intermediate_steps["Conflict_Resolution_Attempt"] = None

                # Append the current persona's turn to history with its final status
                debate_history.append(
                    {
                        "persona": persona_name,
                        "output": turn_output,
                        "status": turn_status,
                    }
                )

                # Update previous_output_for_llm for the next turn
                if turn_status == "corrected":
                    previous_output_for_llm = (
                        turn_output  # Use the corrected output for the next turn
                    )
                elif turn_status == "problematic":
                    # If problematic and not corrected, use the problematic output for the next turn's context
                    previous_output_for_llm = turn_output
                    # If content misalignment was detected, add a corrective note for the next persona
                    if isinstance(turn_output, dict) and turn_output.get(
                        "content_misalignment_warning"
                    ):
                        if isinstance(previous_output_for_llm, str):
                            previous_output_for_llm += "\n\nCRITICAL FEEDBACK: The previous turn was misaligned with the core topic. Ensure your response is strictly focused on the initial prompt and debate objectives."
                        elif isinstance(previous_output_for_llm, dict):
                            # If it's a dict, add a specific key for critical feedback
                            previous_output_for_llm["CRITICAL_FEEDBACK"] = (
                                "The previous turn was misaligned with the core topic. Ensure your response is strictly focused on the initial prompt and debate objectives."
                            )
                else:  # success
                    previous_output_for_llm = turn_output

            except (
                LLMProviderRequestError,
                LLMProviderResponseError,
                CircuitBreakerError,
            ) as e:
                error_output = {
                    "error": f"Turn failed for {persona_name}: {str(e)}",
                    "malformed_blocks": [
                        {"type": "DEBATE_TURN_ERROR", "message": str(e)}
                    ],
                }
                # Append the failed turn to history immediately
                debate_history.append(
                    {
                        "persona": persona_name,
                        "output": error_output,
                        "status": "failed",
                    }
                )  # Mark as failed
                self._log_with_context(
                    "error",
                    f"Non-retryable provider error during {persona_name} turn: {e}. Skipping conflict resolution for this turn. (Raw LLM Output: {raw_llm_output_for_exception[:200]})",  # Use the initialized variable
                    exc_info=True,
                    original_exception=e,
                )
                previous_output_for_llm = error_output
                continue  # Continue to next persona, no conflict resolution for non-retryable errors
            except Exception as e:
                error_output = {
                    "error": f"Turn failed for {persona_name}: {str(e)}",
                    "malformed_blocks": [
                        {"type": "DEBATE_TURN_ERROR", "message": str(e)}
                    ],
                }
                # Append the problematic turn to history first, then try to resolve
                debate_history.append(
                    {
                        "persona": persona_name,
                        "output": error_output,
                        "status": "failed",
                    }
                )
                self._log_with_context(
                    "error",
                    f"Error during {persona_name} turn: {e}. Attempting conflict resolution. ",
                    exc_info=True,
                    original_exception=e,
                )
                resolved_output_from_manager = self.conflict_manager.resolve_conflict(
                    debate_history  # Pass the history including the failed turn
                )
                if resolved_output_from_manager and resolved_output_from_manager.get(
                    "resolved_output"
                ):
                    debate_history.append(
                        {
                            "persona": "Conflict_Resolution_Manager",
                            "output": resolved_output_from_manager,
                            "status": "success",  # Conflict manager's output is considered successful
                        }
                    )
                    previous_output_for_llm = resolved_output_from_manager[
                        "resolved_output"
                    ]
                    self.intermediate_steps["Conflict_Resolution_Attempt"] = {
                        "conflict_resolved": True,
                        "resolution_summary": resolved_output_from_manager[
                            "resolution_summary"
                        ],
                        "resolved_output": resolved_output_from_manager[
                            "resolved_output"
                        ],
                    }
                    self.intermediate_steps["Unresolved_Conflict"] = None
                else:
                    self._log_with_context(
                        "error",
                        f"Conflict resolution failed for {persona_name} error: {e}",
                        persona=persona_name,
                        exc_info=True,
                        original_exception=e,
                    )
                    self.rich_console.print(
                        f"[red]Error during {persona_name} turn: {e}[/red]"
                    )
                    previous_output_for_llm = error_output
                    self.intermediate_steps["Unresolved_Conflict"] = None
                    self.intermediate_steps["Conflict_Resolution_Attempt"] = None
                continue

        self._log_with_context("info", "All debate turns completed.")
        return debate_history

    def _summarize_critical_config_deployment(
        self, summarized_metrics: dict[str, Any], max_tokens: int
    ) -> tuple[dict[str, Any], int]:
        """Summarizes critical configuration and deployment sections."""
        CRITICAL_SECTION_TOKEN_BUDGET = max(
            500, min(int(max_tokens * 0.25), int(max_tokens * 0.3))
        )

        critical_sections_content = {}
        if "configuration_analysis" in summarized_metrics:
            critical_sections_content["configuration_analysis"] = summarized_metrics[
                "configuration_analysis"
            ]
            del summarized_metrics["configuration_analysis"]
        if "deployment_robustness" in summarized_metrics:
            critical_sections_content["deployment_robustness"] = summarized_metrics[
                "deployment_robustness"
            ]
            del summarized_metrics["deployment_robustness"]

        summarized_critical_sections = {}
        critical_sections_tokens = self.tokenizer.count_tokens(
            safe_json_dumps(critical_sections_content, default=convert_to_json_friendly)
        )

        if critical_sections_tokens > CRITICAL_SECTION_TOKEN_BUDGET:
            self._log_with_context(
                "debug",
                "Critical sections (config/deployment) exceed dedicated budget. Summarizing aggressively.",
            )
            for key, content in critical_sections_content.items():
                if key == "configuration_analysis" and content:
                    summarized_config = {
                        "ci_workflow_present": bool(content.get("ci_workflow")),
                        "pre_commit_hooks_count": len(
                            content.get("pre_commit_hooks", [])
                        ),
                        "pyproject_toml_present": bool(content.get("pyproject_toml")),
                        "malformed_blocks_count": len(
                            content.get("malformed_blocks", [])
                        ),
                    }
                    summarized_critical_sections[key] = summarized_config
                elif key == "deployment_robustness" and content:
                    summarized_deployment = {
                        "dockerfile_present": content.get("dockerfile_present"),
                        "dockerfile_healthcheck_present": content.get(
                            "dockerfile_healthcheck_present"
                        ),
                        "dockerfile_non_root_user": content.get(
                            "dockerfile_non_root_user"
                        ),
                        "prod_dependency_count": content.get("prod_dependency_count"),
                        "unpinned_prod_dependencies_count": len(
                            content.get("unpinned_prod_dependencies", [])
                        ),
                        "malformed_blocks_count": len(
                            content.get("malformed_blocks", [])
                        ),
                    }
                    summarized_critical_sections[key] = summarized_deployment
                else:
                    summarized_critical_sections[key] = (
                        f"High-level summary of {key} (truncated)."
                    )
        else:
            summarized_critical_sections = critical_sections_content

        summarized_metrics.update(summarized_critical_sections)
        current_tokens = self.tokenizer.count_tokens(
            safe_json_dumps(summarized_metrics, default=convert_to_json_friendly)
        )
        return summarized_metrics, current_tokens

    def _truncate_detailed_issue_lists(
        self, summarized_metrics: dict[str, Any], remaining_budget_for_issues: int
    ) -> dict[str, Any]:
        """Truncates detailed issue lists within code_quality to fit budget."""
        TOKENS_PER_ISSUE_ESTIMATE = 100
        ISSUE_DETAIL_TOKEN_LIMIT = 50

        for issue_list_key in [
            "detailed_issues",
            "ruff_violations",
            "detailed_security_issues",
        ]:
            if (
                "code_quality" in summarized_metrics
                and issue_list_key in summarized_metrics["code_quality"]
                and summarized_metrics["code_quality"][issue_list_key]
            ):
                original_issues = list(
                    summarized_metrics["code_quality"][issue_list_key]
                )

                num_issues_to_keep = (
                    int(remaining_budget_for_issues / TOKENS_PER_ISSUE_ESTIMATE)
                    if TOKENS_PER_ISSUE_ESTIMATE > 0
                    else 0
                )

                num_issues_to_keep = max(
                    0, min(len(original_issues), num_issues_to_keep)
                )

                truncated_issues = []
                for issue in original_issues[:num_issues_to_keep]:
                    issue_copy = issue.copy()
                    if issue_copy.get("message"):
                        issue_copy["message"] = self.tokenizer.truncate_to_token_limit(
                            issue_copy["message"], ISSUE_DETAIL_TOKEN_LIMIT
                        )
                    if issue_copy.get("code_snippet"):
                        issue_copy["code_snippet"] = (
                            self.tokenizer.truncate_to_token_limit(
                                issue_copy["code_snippet"], ISSUE_DETAIL_TOKEN_LIMIT
                            )
                        )
                    truncated_issues.append(issue_copy)

                if num_issues_to_keep < len(original_issues):
                    if num_issues_to_keep > 0:
                        summarized_metrics["code_quality"][issue_list_key] = (
                            truncated_issues
                        )
                        summarized_metrics["code_quality"][issue_list_key].append(
                            {
                                "type": "TRUNCATION_SUMMARY",
                                "message": f"Only top {num_issues_to_keep} {issue_list_key} are listed due to token limits. Total: {len(original_issues)}.",
                            }
                        )
                    else:
                        del summarized_metrics["code_quality"][issue_list_key]
                        self._log_with_context(
                            "debug",
                            f"Removed {issue_list_key} entirely due to lack of budget.",
                        )
                    self._log_with_context(
                        "debug",
                        f"Truncated {issue_list_key} from {len(original_issues)} to {num_issues_to_keep}.",
                    )
                elif issue_list_key in summarized_metrics["code_quality"]:
                    summarized_metrics["code_quality"][issue_list_key] = (
                        truncated_issues
                    )
            elif (
                "code_quality" in summarized_metrics
                and issue_list_key in summarized_metrics["code_quality"]
                and not summarized_metrics["code_quality"][issue_list_key]
            ):
                del summarized_metrics["code_quality"][issue_list_key]

        if (
            "security" in summarized_metrics
            and "detailed_security_issues" in summarized_metrics["security"]
            and summarized_metrics["security"]["detailed_security_issues"]
        ):
            original_issues = list(
                summarized_metrics["security"]["detailed_security_issues"]
            )
            num_issues_to_keep = (
                int(remaining_budget_for_issues / TOKENS_PER_ISSUE_ESTIMATE)
                if TOKENS_PER_ISSUE_ESTIMATE > 0
                else 0
            )
            num_issues_to_keep = max(0, min(len(original_issues), num_issues_to_keep))

            truncated_issues = []
            for issue in original_issues[:num_issues_to_keep]:
                issue_copy = issue.copy()
                if issue_copy.get("message"):
                    issue_copy["message"] = self.tokenizer.truncate_to_token_limit(
                        issue_copy["message"], ISSUE_DETAIL_TOKEN_LIMIT
                    )
                if issue_copy.get("code_snippet"):
                    issue_copy["code_snippet"] = self.tokenizer.truncate_to_token_limit(
                        issue_copy["code_snippet"], ISSUE_DETAIL_TOKEN_LIMIT
                    )
                truncated_issues.append(issue_copy)

            if num_issues_to_keep < len(original_issues):
                if num_issues_to_keep > 0:
                    summarized_metrics["security"]["detailed_security_issues"] = (
                        truncated_issues
                    )
                    summarized_metrics["security"]["detailed_security_issues"].append(
                        {
                            "type": "TRUNCATION_SUMMARY",
                            "message": f"Only top {num_issues_to_keep} detailed_security_issues are listed due to token limits. Total: {len(original_issues)}.",
                        }
                    )
                else:
                    del summarized_metrics["security"]["detailed_security_issues"]
            elif "detailed_security_issues" in summarized_metrics["security"]:
                summarized_metrics["security"]["detailed_security_issues"] = (
                    truncated_issues
                )

        return summarized_metrics

    def _create_fallback_metrics_summary_string(
        self, summarized_metrics: dict[str, Any], max_tokens: int
    ) -> dict[str, Any]:
        """Creates a high-level summary string if metrics are still too large."""
        self._log_with_context(
            "warning",
            "Metrics still too large after aggressive truncation. Converting to high-level summary string.",
        )
        perf_metrics = summarized_metrics.get("performance_efficiency", {})
        token_usage = perf_metrics.get("token_usage_stats", {})

        total_tokens = token_usage.get("total_tokens", "N/A")
        total_cost_usd = token_usage.get("total_cost_usd", 0.0)

        security_metrics = summarized_metrics.get("security", {})
        high_severity_issues = security_metrics.get("high_severity_issues", 0)
        medium_severity_issues = security_metrics.get("medium_severity_issues", 0)

        code_quality_metrics = summarized_metrics.get("code_quality", {})
        ruff_issues = code_quality_metrics.get("ruff_issues_count", 0)
        maintainability_index = code_quality_metrics.get("maintainability_index", "N/A")

        summary_str = (
            f"Tokens: {total_tokens}, Cost: ${total_cost_usd:.4f}. "
            f"Security: {high_severity_issues} high, {medium_severity_issues} medium severity issues. "
            f"Code Quality: {ruff_issues} Ruff issues, Maintainability Index: {maintainability_index}."
        )
        trimmed_summary_str = self.tokenizer.truncate_to_token_limit(
            summary_str, max(100, int(max_tokens * 0.5))
        )
        return {"summary_string": trimmed_summary_str}

    def _summarize_metrics_for_llm(
        self, metrics: dict[str, Any], max_tokens: int
    ) -> dict[str, Any]:
        """Intelligently summarizes the metrics dictionary to fit within a token budget.
        Prioritizes high-level summaries and truncates verbose lists like 'detailed_issues'.
        Ensures critical configuration and deployment analysis are preserved.
        """
        serializable_metrics = convert_to_json_friendly(metrics)

        summarized_metrics = safe_json_loads(safe_json_dumps(serializable_metrics))

        current_tokens = self.tokenizer.count_tokens(
            safe_json_dumps(summarized_metrics, default=convert_to_json_friendly)
        )

        if current_tokens <= max_tokens:
            return summarized_metrics

        self._log_with_context(
            "debug",
            f"Summarizing metrics for LLM. Current tokens: {current_tokens}, Max: {max_tokens}",
        )

        summarized_metrics, current_tokens = self._summarize_critical_config_deployment(
            summarized_metrics,
            int(
                max_tokens * 0.3
            ),  # Dedicate a portion of budget for critical config/deployment
        )
        # After summarizing critical sections, re-calculate current tokens
        current_tokens = self.tokenizer.count_tokens(
            safe_json_dumps(summarized_metrics, default=convert_to_json_friendly)
        )

        # Prioritize top N issues if still over budget
        remaining_budget_for_issues = max_tokens - current_tokens
        if remaining_budget_for_issues > 0:
            summarized_metrics = self._truncate_detailed_issue_lists(
                summarized_metrics, remaining_budget_for_issues
            )
        current_tokens = self.tokenizer.count_tokens(
            safe_json_dumps(summarized_metrics, default=convert_to_json_friendly)
        )

        if current_tokens > max_tokens:
            return self._create_fallback_metrics_summary_string(
                summarized_metrics, max_tokens
            )

        return summarized_metrics

    def _summarize_debate_history_for_llm(
        self,
        debate_history: list[dict[str, Any]],
        max_tokens: int,
        initial_prompt: str = "",
    ) -> str:  # MODIFIED: Return type is now str, not List[Dict]
        """Summarizes the debate history to fit within a token budget.
        Prioritizes recent turns and concise summaries of each turn's output.
        """
        # Use summarizer pipeline for longer turns if available and budget allows
        # Prioritize recent turns, but summarize their content aggressively
        MAX_RECENT_TURNS = 3
        SUMMARY_TOKEN_LIMIT_PER_TURN = max(50, int(max_tokens / MAX_RECENT_TURNS / 2))

        # Summarize recent turns individually
        recent_turns_summaries = []
        for turn in reversed(debate_history[-MAX_RECENT_TURNS:]):
            turn_copy = copy.deepcopy(turn)
            output_content_str = ""
            if isinstance(turn_copy.get("output"), dict):
                output_content_str = (
                    turn_copy["output"].get("CRITIQUE_SUMMARY", "")
                    or turn_copy["output"].get("ANALYSIS_SUMMARY", "")
                    or turn_copy["output"].get("general_output", "")
                    or turn_copy["output"].get("summary", "")
                )
            elif isinstance(turn_copy.get("output"), str):
                output_content_str = turn_copy["output"]

            if output_content_str:
                # Use summarizer pipeline if available and content is long
                if (
                    self.prompt_optimizer.summarizer_pipeline
                    and self.prompt_optimizer._count_tokens_robustly(output_content_str)
                    > SUMMARY_TOKEN_LIMIT_PER_TURN
                ):
                    summary_text = self.prompt_optimizer._summarize_text(
                        output_content_str,
                        SUMMARY_TOKEN_LIMIT_PER_TURN,
                        truncation_indicator="... (summarized)",
                    )
                    turn_copy["output_summary"] = summary_text
                    self._log_with_context(
                        "debug",
                        f"Summarized turn output for {turn_copy.get('persona', 'unknown')}.",
                    )
                else:
                    # Fallback to truncation if summarizer not available or content is short
                    turn_copy["output_summary"] = (
                        self.tokenizer.truncate_to_token_limit(
                            output_content_str,
                            SUMMARY_TOKEN_LIMIT_PER_TURN,
                            truncation_indicator="... (truncated)",
                        )
                    )
            else:
                turn_copy["output_summary"] = "No content provided."

            # Remove original verbose output to save tokens
            turn_copy.pop("output", None)
            recent_turns_summaries.insert(
                0, turn_copy
            )  # Add to the beginning to maintain chronological order

        # Now, combine the summarized recent turns and ensure overall fit
        combined_summary_str = safe_json_dumps(
            recent_turns_summaries, indent=2, default=convert_to_json_friendly
        )
        final_history_summary = self.tokenizer.truncate_to_token_limit(
            combined_summary_str,
            max_tokens,
            truncation_indicator="\n... (older debate history omitted)",
        )

        if not final_history_summary and debate_history:
            self._log_with_context(
                "warning", "Debate history too large, providing minimal summary."
            )
            return f"Debate history contains {len(debate_history)} turns. Too verbose to include in full. Only recent turns summarized."
        return final_history_summary

    def _perform_synthesis_phase(
        self, persona_sequence: list[str], debate_persona_results: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], Optional[FocusedMetricsCollector]]:
        """Executes the final synthesis persona turn based on the debate history."""
        self._log_with_context("info", "Executing final synthesis persona turn.")

        synthesis_persona_name = ""
        if self.persona_manager.prompt_analyzer.is_self_analysis_prompt(
            self.initial_prompt
        ):
            synthesis_persona_name = "Self_Improvement_Analyst"
        elif self.domain == "Software Engineering":
            synthesis_persona_name = "Impartial_Arbitrator"
        else:
            synthesis_persona_name = "General_Synthesizer"

        synthesis_persona_config = self.persona_manager.get_adjusted_persona_config(
            synthesis_persona_name
        )
        if not synthesis_persona_config:
            self._log_with_context(
                "error",
                f"Synthesis persona '{synthesis_persona_name}' configuration not found. Falling back to General_Synthesizer.",
            )
            synthesis_persona_name = "General_Synthesizer"
            synthesis_persona_config = self.persona_manager.get_adjusted_persona_config(
                synthesis_persona_name
            )
            if not synthesis_persona_config:
                raise ChimeraError("No synthesis persona configuration found.")

        self.synthesis_persona_name_for_metrics = synthesis_persona_name

        local_metrics_collector: Optional[FocusedMetricsCollector] = None

        # MODIFIED: Removed manual prompt_parts construction, now using Jinja2 template
        # synthesis_prompt_parts = [f"Initial Problem: {self.initial_prompt}\n\n"]

        files_for_synthesis_context = []
        if self.raw_file_contents:
            # MODIFIED: Limit the file list passed to the synthesis prompt
            # Use relevant_files from context_analysis_results if available, otherwise truncate raw_file_contents keys
            context_analysis_results = self.intermediate_steps.get(
                "Context_Analysis_Output"
            )
            if context_analysis_results and context_analysis_results.get(
                "relevant_files"
            ):
                # context_analysis_results['relevant_files'] is a list of (file_path, score) tuples
                files_for_synthesis_context = [
                    f[0] for f in context_analysis_results["relevant_files"]
                ]
                self._log_with_context(
                    "debug",
                    f"Using {len(files_for_synthesis_context)} relevant files from context analysis for synthesis prompt.",
                )
            else:
                # Fallback: use a truncated list of all files if context analysis didn't run or found nothing
                all_files = list(self.raw_file_contents.keys())
                MAX_FILES_IN_SYNTHESIS_LIST = 50  # Limit to 50 files for the list
                files_for_synthesis_context = sorted(all_files)[
                    :MAX_FILES_IN_SYNTHESIS_LIST
                ]
                if len(all_files) > MAX_FILES_IN_SYNTHESIS_LIST:
                    self._log_with_context(
                        "warning",
                        f"Truncated file list for synthesis prompt from {len(all_files)} to {MAX_FILES_IN_SYNTHESIS_LIST} files.",
                    )

        file_list_json_str = safe_json_dumps(files_for_synthesis_context, indent=2)

        if (
            self.is_self_analysis
            and synthesis_persona_name == "Self_Improvement_Analyst"
            and not self.raw_file_contents
        ):
            self._log_with_context(
                "warning",
                "Self-Improvement Analyst cannot perform detailed analysis without codebase access. Returning specific error message.",
            )
            error_output = SelfImprovementAnalysisOutputV1(
                ANALYSIS_SUMMARY="Project Chimera's codebase analysis requires access to the codebase itself. Without the codebase, I cannot identify specific areas for improvement, provide actionable code modifications, or apply the 80/20 principle to enhance reasoning quality, robustness, efficiency, or maintainability. To proceed, please provide the Project Chimera codebase or specific files/modules you would like analyzed.",
                IMPACTFUL_SUGGESTIONS=[],
                malformed_blocks=[
                    {
                        "type": "CODEBASE_ACCESS_REQUIRED",
                        "message": "Self-Improvement Analyst requires codebase access for detailed analysis.",
                    }
                ],
            ).model_dump(by_alias=True)

            self.persona_manager.record_persona_performance(
                synthesis_persona_name,
                1,
                error_output,
                False,
                "Missing codebase context for self-analysis",
                is_truncated=False,
                schema_validation_failed=False,
                token_budget_exceeded=False,
                tokens_used_in_turn=0,
                is_successful_turn=False,
            )
            return error_output, None

        # Summarize the debate history instead of passing the full verbose JSON
        debate_history_summary_budget = int(self.phase_budgets["synthesis"] * 0.1)
        summarized_debate_history = self._summarize_debate_history_for_llm(
            debate_persona_results,
            max(
                200,
                min(
                    debate_history_summary_budget, self.phase_budgets["synthesis"] // 4
                ),
            ),
            self.initial_prompt,
        )

        conflict_resolution_summary = None
        if self.intermediate_steps.get("Conflict_Resolution_Attempt"):
            conflict_resolution_summary = safe_json_dumps(
                self.intermediate_steps["Conflict_Resolution_Attempt"][
                    "resolution_summary"
                ],
                indent=2,
                default=convert_to_json_friendly,
            )

        unresolved_conflict = None
        if self.intermediate_steps.get("Unresolved_Conflict"):
            unresolved_conflict = safe_json_dumps(
                self.intermediate_steps["Unresolved_Conflict"],
                indent=2,
                default=convert_to_json_friendly,
            )

        metrics_json_str = ""
        if synthesis_persona_name == "Self_Improvement_Analyst":
            local_metrics_collector = FocusedMetricsCollector(
                initial_prompt=self.initial_prompt,
                debate_history=list(debate_persona_results),
                intermediate_steps=self.intermediate_steps,
                tokenizer=self.tokenizer,
                llm_provider=self.llm_provider,
                persona_manager=self.persona_manager,
                content_validator=self.content_validator,
                codebase_scanner=self.codebase_scanner,
                token_tracker=self.token_tracker,  # NEW: Pass token_tracker to metrics_collector
            )
            local_metrics_collector.collect_all_metrics()
            self.file_analysis_cache = local_metrics_collector.file_analysis_cache

            effective_metrics_budget = max(
                300,
                min(
                    int(self.phase_budgets["synthesis"] * 0.3),
                    self.phase_budgets["synthesis"] // 3,
                ),
            )
            summarized_metrics = self._summarize_metrics_for_llm(
                local_metrics_collector.collected_metrics, effective_metrics_budget
            )
            self.intermediate_steps["Self_Improvement_Metrics"] = summarized_metrics
            metrics_json_str = safe_json_dumps(
                summarized_metrics, indent=2, default=convert_to_json_friendly
            )

        # MODIFIED: Use PromptOptimizer to generate the final synthesis prompt
        synthesis_template_context = {
            "initial_prompt": self.initial_prompt,
            "debate_history_summary": summarized_debate_history,
            "metrics_json_str": metrics_json_str,
            "file_list_json_str": file_list_json_str,
            "conflict_resolution_summary": conflict_resolution_summary,
            "unresolved_conflict": unresolved_conflict,
        }

        template_name_to_use = synthesis_persona_name.lower()

        final_synthesis_prompt_raw = self.prompt_optimizer.generate_prompt(
            template_name=template_name_to_use, context=synthesis_template_context
        )

        input_budget_for_synthesis_prompt = int(self.phase_budgets["synthesis"] * 0.4)

        final_synthesis_prompt = self.tokenizer.truncate_to_token_limit(
            final_synthesis_prompt_raw,
            input_budget_for_synthesis_prompt,
            truncation_indicator="\n... (truncated for token limits) ...",
        )

        max_output_tokens_for_turn = self.phase_budgets[
            "synthesis"
        ] - self.tokenizer.count_tokens(final_synthesis_prompt)
        max_output_tokens_for_turn = max(
            500, min(max_output_tokens_for_turn, synthesis_persona_config.max_tokens)
        )

        estimated_tokens_for_turn = (
            self.tokenizer.count_tokens(final_synthesis_prompt)
            + max_output_tokens_for_turn
        )
        self.check_budget(
            "synthesis", estimated_tokens_for_turn, synthesis_persona_name
        )

        # NEW: Log token usage for 80/20 analysis
        perf_logger = get_performance_logger()
        perf_logger.log_token_usage(
            estimated_tokens_for_turn, self.max_total_tokens_budget, self.request_id
        )

        synthesis_output = self._execute_llm_turn(
            synthesis_persona_name,
            final_synthesis_prompt,
            "synthesis",
            max_output_tokens_for_turn,
            context_for_template=synthesis_template_context,
        )
        self._log_with_context("info", "Final synthesis persona turn completed.")

        if (
            self.synthesis_persona_name_for_metrics == "Self_Improvement_Analyst"
            and local_metrics_collector
        ):
            local_metrics_collector.analyze_reasoning_quality(synthesis_output)
            self.intermediate_steps["Self_Improvement_Metrics"] = (
                local_metrics_collector.collected_metrics
            )
            is_successful_suggestion = not self._is_problematic_output(synthesis_output)
            local_metrics_collector.record_self_improvement_suggestion_outcome(
                self.synthesis_persona_name_for_metrics,
                is_successful_suggestion,
                schema_failed=bool(synthesis_output.get("malformed_blocks")),
            )

        return (synthesis_output, local_metrics_collector)

    def _finalize_debate_results(
        self,
        context_persona_turn_results: Optional[dict[str, Any]],
        debate_persona_results: list[dict[str, Any]],
        synthesis_persona_results: Optional[dict[str, Any]],
    ) -> tuple[Any, dict[str, Any]]:
        """Synthesizes the final answer and prepares the intermediate steps for display."""
        final_answer = synthesis_persona_results

        if not isinstance(final_answer, dict):
            final_answer = {"general_output": str(final_answer), "malformed_blocks": []}
        if "malformed_blocks" not in final_answer:
            final_answer["malformed_blocks"] = []

        if isinstance(final_answer, dict):
            if final_answer.get("version") == "1.0" and "data" in final_answer:
                final_answer["data"] = self._consolidate_self_improvement_code_changes(
                    final_answer["data"]
                )
                self._log_with_context(
                    "info",
                    "Self-improvement code changes consolidated (versioned output).",
                )
            elif (
                "ANALYSIS_SUMMARY" in final_answer
                and "IMPACTFUL_SUGGESTIONS" in final_answer
            ):
                v1_data_consolidated = self._consolidate_self_improvement_code_changes(
                    final_answer
                )
                final_answer = SelfImprovementAnalysisOutput(
                    version="1.0",
                    data=v1_data_consolidated,
                    malformed_blocks=final_answer.get("malformed_blocks", []),
                ).model_dump(by_alias=True)
                self._log_with_context(
                    "info",
                    "Self-improvement code changes consolidated and wrapped (direct V1 output).",
                )

        self._update_intermediate_steps_with_totals()
        if "malformed_blocks" not in self.intermediate_steps:
            self.intermediate_steps["malformed_blocks"] = []

        if self.intermediate_steps.get("Conflict_Resolution_Attempt"):
            final_answer["CONFLICT_RESOLUTION"] = self.intermediate_steps[
                "Conflict_Resolution_Attempt"
            ]["resolution_summary"]
            final_answer["UNRESOLVED_CONFLICT"] = None
        elif self.intermediate_steps.get("Unresolved_Conflict"):
            final_answer["UNRESOLVED_CONFLICT"] = (
                self.intermediate_steps["Unresolved_Conflict"]["summary"]
                if isinstance(self.intermediate_steps["Unresolved_Conflict"], dict)
                and "summary" in self.intermediate_steps["Unresolved_Conflict"]
                else str(self.intermediate_steps["Unresolved_Conflict"])
            )
            final_answer["CONFLICT_RESOLUTION"] = None

        return final_answer, self.intermediate_steps

    def _is_problematic_output(self, output: dict[str, Any]) -> bool:
        """Checks if a persona's output is malformed, indicates content misalignment, or reports a conflict."""
        if not isinstance(output, dict):
            return True
        if output.get("malformed_blocks") or output.get("content_misalignment_warning"):
            return True
        return bool("conflict_found" in output and output["conflict_found"] is True)

    def _calculate_pareto_score(self, finding: Any) -> float:
        """Calculate 80/20 Pareto score for a finding (impact/effort)."""
        impact = (finding.metrics.expected_quality_improvement or 0) + (
            finding.metrics.token_savings_percent or 0
        )
        effort_factor = 1.0 / (finding.metrics.estimated_effort or 1)
        return lru_cache(maxsize=128)(lambda: min(1.0, impact * effort_factor * 5))()

    def _update_intermediate_steps_with_totals(self):
        """Updates the intermediate steps dictionary with total token usage and estimated cost."""
        self.intermediate_steps["Total_Tokens_Used"] = self.token_tracker.current_usage
        self.intermediate_steps["Total_Estimated_Cost_USD"] = (
            self.get_total_estimated_cost()
        )

    def _handle_codebase_access_error(
        self, e: CodebaseAccessError
    ) -> tuple[Any, dict[str, Any]]:
        """Handles the CodebaseAccessError by generating a specific, actionable output."""
        self._log_with_context("error", f"Codebase access error: {e}", exc_info=True)

        # Dynamically generate the list of critical files needed for the proposed solution
        critical_files_list_str = ", ".join(CRITICAL_FILES_FOR_SELF_ANALYSIS)

        error_output = SelfImprovementAnalysisOutputV1(
            ANALYSIS_SUMMARY=f"Codebase access required for meaningful self-improvement analysis. {e}",
            IMPACTFUL_SUGGESTIONS=[
                SuggestionItem(
                    AREA="Codebase Access",
                    PROBLEM="No source code files provided or accessible for analysis.",
                    PROPOSED_SOLUTION=f"Provide access to the following critical files: {critical_files_list_str}. Ensure the application has read permissions to the project directory and that the scanner is configured correctly (e.g., include/exclude patterns). Refer to `docs/project_chimera_context.md` for detailed instructions.",
                    EXPECTED_IMPACT="Enables comprehensive critique and identification of architectural improvements related to reasoning, robustness, efficiency, maintainability, security, test coverage, CI/CD, and token usage.",
                    CODE_CHANGES_SUGGESTED=[
                        {
                            "FILE_PATH": "docs/project_chimera_context.md",
                            "ACTION": "MODIFY",
                            "DIFF_CONTENT": "--- a/docs/project_chimera_context.md\n+++ b/docs/project_chimera_context.md\n@@ -1,5 +1,22 @@\n # Project Chimera Codebase Context\n \n This document outlines the necessary structure and files required for effective self-improvement analysis.\n+\n+## Required Files for Analysis\n+- `core.py`: Main application logic\n+- `app.py`: Streamlit web application entry point\n+- `src/llm_provider.py`: LLM interaction and API handling\n+- `src/services.py`: Core service implementations\n+- `src/self_improvement/critique_engine.py`: Critique and analysis logic\n+- `src/self_improvement/improvement_applicator.py`: Application of improvements\n+- `src/self_improvement/metrics_collector.py`: Metrics collection for self-improvement\n+- `src/config/settings.py`: Application settings\n+- `src/constants.py`: Global constants\n+- `docs/project_chimera_context.md`: This documentation file itself\n+- `schemas/analysis_schema.json`: Schema for self-improvement output\n+- `personas.yaml`: Persona definitions\n+- `pyproject.toml`: Project configuration\n+- `.github/workflows/analysis.yml`: Self-analysis workflow\n+- `.github/workflows/ci.yml`: CI workflow\n+- `.ruff.toml`: Ruff linter/formatter configuration\n+- `.pre-commit-config.yaml`: Pre-commit hooks configuration\n+- `requirements.txt`: Development Python dependencies\n+- `requirements-prod.txt`: Production Python dependencies\n+- `Dockerfile`: Docker container definition\n+- All other files in `src/` and `tests/` directories\n+\n+## How to Provide Context\n+When running self-improvement analysis, ensure these files are included in the context provided to the analysis system. The context should include:\n+- File paths\n+- File contents\n+- Directory structure\n+- Any relevant configuration files\n+\n+This will enable the system to perform a thorough analysis and identify meaningful improvements.\n",
                        }
                    ],
                )
            ],
            malformed_blocks=[{"type": "CODEBASE_ACCESS_REQUIRED", "message": str(e)}],
        ).model_dump(by_alias=True)
        self._update_intermediate_steps_with_totals()
        return error_output, self.intermediate_steps

    @handle_errors(default_return=None, log_level="ERROR")
    def run_debate(self) -> tuple[Any, dict[str, Any]]:
        """Orchestrates the full Socratic debate process by calling phase-specific methods.
        Returns the final synthesized answer and a dictionary of intermediate steps.
        """
        self.debate_start_time = datetime.now()
        debate_success = True

        try:
            self._initialize_debate_state()

            # Check for codebase access before proceeding with self-analysis
            if self.is_self_analysis and (
                not self.raw_file_contents
                or all(not content for content in self.raw_file_contents.values())
            ):
                raise CodebaseAccessError(
                    f"No codebase content available for self-analysis. "
                    f"Attempted project root: {self.codebase_scanner.project_root if self.codebase_scanner else 'N/A'}. "
                    "Ensure relevant files are present, not empty, and the scanner is configured correctly (e.g., include/exclude patterns)."
                )

            initial_persona_sequence = self._get_final_persona_sequence(
                self.initial_prompt, None
            )
            self.intermediate_steps["Persona_Sequence_Initial"] = (
                initial_persona_sequence
            )

            context_analysis_results = self._perform_context_analysis_phase(
                tuple(initial_persona_sequence)
            )
            self.intermediate_steps["Context_Analysis_Output"] = (
                context_analysis_results
            )

            persona_sequence = self._get_final_persona_sequence(
                self.initial_prompt, context_analysis_results
            )
            persona_sequence = (
                self.persona_manager.get_token_optimized_persona_sequence(
                    persona_sequence
                )
            )
            self.intermediate_steps["Persona_Sequence"] = persona_sequence

            self._distribute_debate_persona_budgets(persona_sequence)

            context_persona_turn_results = None
            if "Context_Aware_Assistant" in persona_sequence:
                self.status_callback(
                    "Phase 2: Context-Aware Assistant Turn...",
                    state="running",
                    current_total_tokens=self.token_tracker.current_usage,
                    current_total_cost=self.get_total_estimated_cost(),
                    progress_pct=self.get_progress_pct("debate"),
                    current_persona_name="Context_Aware_Assistant",
                )
                context_persona_turn_results = self._process_context_persona_turn(
                    persona_sequence, context_analysis_results
                )
                self.intermediate_steps["Context_Aware_Assistant_Output"] = (
                    context_persona_turn_results
                )

            self.status_callback(
                "Phase 3: Executing Debate Turns...",
                state="running",
                current_total_tokens=self.token_tracker.current_usage,
                current_total_cost=self.get_total_estimated_cost(),
                progress_pct=self.get_progress_pct("debate"),
            )
            debate_persona_results = self._execute_debate_persona_turns(
                persona_sequence,
                context_persona_turn_results
                if context_persona_turn_results is not None
                else {},
            )
            self.intermediate_steps["Debate_History"] = debate_persona_results

            self.status_callback(
                "Phase 4: Synthesizing Final Answer...",
                state="running",
                current_total_tokens=self.token_tracker.current_usage,
                current_total_cost=self.get_total_estimated_cost(),
                progress_pct=self.get_progress_pct("synthesis"),
            )

            (
                synthesis_persona_results,
                metrics_collector_instance_from_synthesis_phase,
            ) = self._perform_synthesis_phase(persona_sequence, debate_persona_results)
            self.intermediate_steps["Final_Synthesis_Output"] = (
                synthesis_persona_results
            )

            if (
                self.synthesis_persona_name_for_metrics == "Self_Improvement_Analyst"
                and metrics_collector_instance_from_synthesis_phase
            ):
                is_successful_suggestion = not self._is_problematic_output(
                    synthesis_persona_results
                )
                metrics_collector_instance_from_synthesis_phase.record_self_improvement_suggestion_outcome(
                    self.synthesis_persona_name_for_metrics,
                    is_successful_suggestion,
                    schema_failed=bool(
                        synthesis_persona_results.get("malformed_blocks")
                    ),
                )

            self.status_callback(
                "Finalizing Results...",
                state="running",
                current_total_tokens=self.token_tracker.current_usage,
                current_total_cost=self.get_total_estimated_cost(),
                progress_pct=0.95,
            )
            final_answer, intermediate_steps = self._finalize_debate_results(
                context_persona_turn_results,
                debate_persona_results,
                synthesis_persona_results,
            )

            self.status_callback(
                "Socratic Debate Complete!",
                state="complete",
                current_total_tokens=self.token_tracker.current_usage,
                current_total_cost=self.get_total_estimated_cost(),
                progress_pct=1.0,
            )
            self._log_with_context(
                "info",
                "Socratic Debate process completed successfully.",
                total_tokens=self.token_tracker.current_usage,
                total_cost=self.get_total_estimated_cost(),
            )

            return final_answer, intermediate_steps
        except CodebaseAccessError as e:
            debate_success = False
            return self._handle_codebase_access_error(e)
        except Exception as e:
            debate_success = False
            self.system_monitor.record_error(type(e).__name__, str(e), self.request_id)
            raise
        finally:
            # Record final debate metrics
            if self.debate_start_time:
                duration = (datetime.now() - self.debate_start_time).total_seconds()
                self.system_monitor.record_debate_timing(duration, self.request_id)
                self.system_monitor.record_metric(
                    MetricType.PERFORMANCE,
                    "debate_total_tokens",
                    self.token_tracker.current_usage if self.token_tracker else 0,
                    context={"debate_id": self.request_id},
                )
                # Record success/failure status
                self.system_monitor.record_metric(
                    MetricType.PERFORMANCE,
                    "debate_success",
                    1.0 if debate_success else 0.0,
                    context={
                        "debate_id": self.request_id,
                        "status": "success" if debate_success else "failed",
                    },
                )
            self.close()

    def _generate_unified_diff(
        self, file_path: str, original_content: str, new_content: str
    ) -> str:
        """Generates a unified diff string between original and new content."""
        diff_lines = difflib.unified_diff(
            original_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
        )
        return "".join(diff_lines)

    def _process_single_file_code_changes(
        self,
        file_path: str,
        changes_for_file: list[CodeChange],
        suggestion: dict[str, Any],
        analysis_output: dict[str, Any],
    ) -> Optional[CodeChange]:
        """Processes and consolidates changes for a single file, with strict file existence validation."""
        file_exists_in_codebase = file_path in self.raw_file_contents
        result_change: Optional[CodeChange] = None  # NEW: Single result variable

        remove_actions = [c for c in changes_for_file if c.action == "REMOVE"]
        add_actions = [c for c in changes_for_file if c.action == "ADD"]
        modify_actions = [c for c in changes_for_file if c.action == "MODIFY"]
        create_directory_actions = [
            c for c in changes_for_file if c.action == "CREATE_DIRECTORY"
        ]

        if create_directory_actions:
            # If there's a CREATE_DIRECTORY action, it takes precedence for that path
            # and other actions for the same path might be invalid or redundant.
            # For simplicity, we'll just return the first CREATE_DIRECTORY action.
            # More complex logic might consolidate multiple directory creations or handle conflicts.
            if file_exists_in_codebase:
                self._log_with_context(
                    "warning",
                    f"CREATE_DIRECTORY action suggested for existing path: {file_path}. Skipping.",
                )
                analysis_output.setdefault("malformed_blocks", []).append(
                    {
                        "type": "INVALID_CREATE_DIRECTORY_ACTION",
                        "message": f"CREATE_DIRECTORY action suggested for existing path '{file_path}'. Suggestion ignored.",
                        "file_path": file_path,
                        "suggestion_area": suggestion.get("AREA"),
                    }
                )
            else:
                result_change = create_directory_actions[0]

        if remove_actions:
            if not file_exists_in_codebase and not any(
                a.action == "ADD" for a in changes_for_file
            ):
                self._log_with_context(
                    "warning",
                    f"REMOVE action suggested for non-existent file: {file_path}. Skipping.",
                )
                analysis_output.setdefault("malformed_blocks", []).append(
                    {
                        "type": "INVALID_REMOVE_ACTION",
                        "message": f"REMOVE action suggested for non-existent file '{file_path}'. Suggestion ignored.",
                        "file_path": file_path,
                        "suggestion_area": suggestion.get("AREA"),
                    }
                )
            else:
                all_lines_to_remove = []
                for ra in remove_actions:
                    all_lines_to_remove.extend(ra.lines)
                result_change = CodeChange(
                    FILE_PATH=file_path,
                    ACTION="REMOVE",
                    LINES=list(set(all_lines_to_remove)),
                )

        if add_actions:
            if file_exists_in_codebase and not any(
                a.action == "REMOVE" for a in changes_for_file
            ):
                self._log_with_context(
                    "warning",
                    f"ADD action suggested for existing file: {file_path}. Converting to MODIFY if content provided.",
                )
                if add_actions[0].full_content:
                    result_change = CodeChange(
                        FILE_PATH=file_path,
                        ACTION="MODIFY",
                        FULL_CONTENT=add_actions[0].full_content,
                        DIFF_CONTENT=self._generate_unified_diff(
                            file_path,
                            self.raw_file_contents.get(file_path, ""),
                            add_actions[0].full_content,
                        ),
                    )
                else:
                    analysis_output.setdefault("malformed_blocks", []).append(
                        {
                            "type": "INVALID_ADD_ACTION",
                            "message": f"ADD action suggested for existing file '{file_path}'. Suggestion ignored.",
                            "file_path": file_path,
                            "suggestion_area": suggestion.get("AREA"),
                        }
                    )
            else:
                result_change = add_actions[0]

        if modify_actions:
            if not file_exists_in_codebase and not any(
                a.action == "ADD" for a in changes_for_file
            ):
                self._log_with_context(
                    "warning",
                    f"MODIFY action suggested for non-existent file: {file_path}. Converting to CREATE if content provided.",
                )
                if modify_actions[0].full_content:
                    result_change = CodeChange(
                        FILE_PATH=file_path,
                        ACTION="CREATE",
                        FULL_CONTENT=modify_actions[0].full_content,
                    )
                else:
                    analysis_output.setdefault("malformed_blocks", []).append(
                        {
                            "type": "INVALID_MODIFY_ACTION",
                            "message": f"MODIFY action suggested for non-existent file '{file_path}'. Suggestion ignored.",
                            "file_path": file_path,
                            "suggestion_area": suggestion.get("AREA"),
                        }
                    )
            else:
                original_content = self.raw_file_contents.get(file_path, "")
                final_content_for_diff = self.raw_file_contents.get(file_path, "")
                last_full_content_provided = None

                for mod_change in modify_actions:
                    if mod_change.full_content is not None:
                        last_full_content_provided = mod_change.full_content

                consolidated_diff_content = None

                if last_full_content_provided is not None:
                    final_content_for_diff = last_full_content_provided
                    consolidated_diff_content = self._generate_unified_diff(
                        file_path, original_content, final_content_for_diff
                    )
                    self._log_with_context(
                        "debug", f"Generated diff from FULL_CONTENT for {file_path}."
                    )
                else:
                    last_diff_from_llm = None
                    for mod_change in modify_actions:
                        if mod_change.diff_content is not None:
                            last_diff_from_llm = mod_change.diff_content

                    if last_diff_from_llm is not None:
                        consolidated_diff_content = last_diff_from_llm
                        self._log_with_context(
                            "debug", f"Using LLM-provided DIFF_CONTENT for {file_path}."
                        )
                    else:
                        self._log_with_context(
                            "info",
                            f"Consolidated MODIFY for {file_path} resulted in no effective change (no FULL_CONTENT or DIFF_CONTENT provided). Removing from suggestions.",
                        )
                        analysis_output.setdefault("malformed_blocks", []).append(
                            {
                                "type": "NO_OP_CODE_CHANGE_CONSOLIDATED",
                                "message": f"Consolidated MODIFY for {file_path} resulted in no effective change. Removed from final suggestions.",
                                "file_path": file_path,
                                "suggestion_area": suggestion.get("AREA"),
                            }
                        )
                        return None
                if consolidated_diff_content and consolidated_diff_content.strip():
                    result_change = CodeChange(
                        FILE_PATH=file_path,
                        ACTION="MODIFY",
                        DIFF_CONTENT=consolidated_diff_content,
                    )
                else:
                    self._log_with_context(
                        "info",
                        f"Consolidated MODIFY for {file_path} resulted in no effective change. Removing from suggestions.",
                        file_path=file_path,
                        suggestion_area=suggestion.get("AREA"),
                    )
                    analysis_output.setdefault("malformed_blocks", []).append(
                        {
                            "type": "NO_OP_CODE_CHANGE_CONSOLIDATED",
                            "message": f"Consolidated MODIFY for {file_path} resulted in no effective change. Removed from final suggestions.",
                            "file_path": file_path,
                            "suggestion_area": suggestion.get("AREA"),
                        }
                    )
                    return None
        return result_change  # FIX PLR0911

    def _consolidate_self_improvement_code_changes(
        self, analysis_output: dict[str, Any]
    ) -> dict[str, Any]:
        """Consolidates multiple CODE_CHANGES_SUGGESTED for the same file within
        SelfImprovementAnalysisOutput. Also filters out no-op changes.
        """
        if "IMPACTFUL_SUGGESTIONS" not in analysis_output:
            return analysis_output

        # Add a check for malformed blocks from path validation and log them
        if analysis_output.get("malformed_blocks"):
            for block in analysis_output["malformed_blocks"]:
                if block.get("type") in [
                    "INVALID_MODIFY_ACTION",
                    "INVALID_ADD_ACTION",
                    "INVALID_REMOVE_ACTION",
                    "INVALID_FILE_PATH_OR_ACTION",
                ]:
                    self._log_with_context(
                        "warning",
                        f"Invalid file action detected during consolidation: {block.get('message')}",
                        file_path=block.get("file_path"),
                        action=block.get("action"),
                        suggestion_area=block.get("suggestion_area"),
                    )

        processed_suggestions = []
        for suggestion_data in analysis_output[
            "IMPACTFUL_SUGGESTIONS"
        ]:  # MODIFIED: Renamed 'suggestion' to 'suggestion_data'
            if (
                "CODE_CHANGES_SUGGESTED"
                not in suggestion_data  # MODIFIED: Used 'suggestion_data'
                or not suggestion_data[
                    "CODE_CHANGES_SUGGESTED"
                ]  # MODIFIED: Used 'suggestion_data'
            ):
                processed_suggestions.append(
                    suggestion_data
                )  # MODIFIED: Used 'suggestion_data'
                continue

            file_changes_map = defaultdict(list)
            malformed_blocks_for_suggestion = []
            for change_data in suggestion_data.get(
                "CODE_CHANGES_SUGGESTED", []
            ):  # MODIFIED: Used 'suggestion_data'
                try:
                    code_change = CodeChange.model_validate(change_data)
                    file_changes_map[code_change.file_path].append(code_change)
                except ValidationError as e:
                    self._log_with_context(
                        "warning",
                        f"Malformed CodeChange item during consolidation: {e}. Skipping.",
                        raw_change_data=change_data,
                    )
                    analysis_output.setdefault("malformed_blocks", []).append(
                        {
                            "type": "CODE_CHANGE_CONSOLIDATION_ERROR",
                            "message": f"Malformed CodeChange item skipped during consolidation: {e}",
                            "raw_string_snippet": str(change_data)[:500],
                        }
                    )
                    continue

            new_code_changes_for_suggestion = []
            for file_path, changes_for_file in file_changes_map.items():
                consolidated_change = self._process_single_file_code_changes(
                    file_path,
                    changes_for_file,
                    suggestion_data,
                    analysis_output,  # MODIFIED: Used 'suggestion_data'
                )
                if consolidated_change:
                    new_code_changes_for_suggestion.append(
                        consolidated_change.model_dump(by_alias=True)
                    )

            # FIX: Corrected typo here
            suggestion_data["CODE_CHANGES_SUGGESTED"] = (
                new_code_changes_for_suggestion  # MODIFIED: Used 'suggestion_data'
            )
            if malformed_blocks_for_suggestion:
                suggestion_data.setdefault(
                    "malformed_blocks", []
                ).extend(  # MODIFIED: Used 'suggestion_data'
                    malformed_blocks_for_suggestion
                )
            processed_suggestions.append(
                suggestion_data
            )  # MODIFIED: Used 'suggestion_data'

        analysis_output["IMPACTFUL_SUGGESTIONS"] = processed_suggestions
        return analysis_output
