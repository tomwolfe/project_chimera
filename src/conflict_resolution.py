import json
import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING

# NEW: Import necessary classes for self-correction
from src.llm_provider import GeminiProvider
from src.models import (
    PersonaConfig,
    GeneralOutput,
    LLMOutput,
    CritiqueOutput,
    ConflictReport,
    SelfImprovementAnalysisOutputV1, # Keep this import for the new error handling
    ContextAnalysisOutput,
    ConfigurationAnalysisOutput,
    DeploymentAnalysisOutput,
    SuggestionItem,  # NEW: Import SuggestionItem for conflict detection
)
from src.utils.output_parser import LLMOutputParser
from src.llm_tokenizers.gemini_tokenizer import GeminiTokenizer
from src.config.settings import ChimeraSettings
from src.constants import SHARED_JSON_INSTRUCTIONS
from src.exceptions import ChimeraError
from pydantic import ValidationError # Import ValidationError for model_validate

# Use TYPE_CHECKING to avoid circular import at runtime
if TYPE_CHECKING:
    from src.persona_manager import PersonaManager

logger = logging.getLogger(__name__)


class ConflictResolutionManager:
    """
    Manages the resolution of conflicts or malformed outputs from LLM personas.
    Implements strategies to attempt to recover or synthesize a coherent output.
    """

    def __init__(
        self,
        llm_provider: Optional[GeminiProvider] = None,
        persona_manager: Optional["PersonaManager"] = None,
    ):
        logger.info("ConflictResolutionManager initialized.")
        self.llm_provider = llm_provider
        self.persona_manager = persona_manager
        self.max_self_correction_retries = 2
        self.settings = ChimeraSettings()
        self.output_parser = LLMOutputParser()

    # NEW METHOD: _detect_conflict_type
    def _detect_conflict_type(self, debate_history: List[Dict]) -> str:
        """Detect the type of conflict between personas."""
        logger.info("Attempting to detect conflict type.")

        # Look for fundamental disagreements in key areas
        security_auditor_output = next(
            (t["output"] for t in debate_history if t["persona"] == "Security_Auditor"),
            None,
        )
        code_architect_output = next(
            (t["output"] for t in debate_history if t["persona"] == "Code_Architect"),
            None,
        )
        devils_advocate_output = next(
            (t["output"] for t in debate_history if t["persona"] == "Devils_Advocate"),
            None,
        )

        # Check for SECURITY_VS_ARCHITECTURE conflict
        if security_auditor_output and code_architect_output:
            # Assuming CritiqueOutput structure for Security_Auditor and Code_Architect
            security_suggestions = security_auditor_output.get("SUGGESTIONS", [])
            architect_suggestions = code_architect_output.get("SUGGESTIONS", [])

            security_problems_identified = [
                s.get("PROBLEM", "").lower()
                for s in security_suggestions
                if s.get("AREA", "").lower() == "security"
            ]
            architect_solutions_proposed = [
                s.get("PROPOSED_SOLUTION", "").lower()
                for s in architect_suggestions
                if s.get("AREA", "").lower() == "architecture"
                or s.get("AREA", "").lower() == "maintainability"
            ]

            # If security issues are identified but not adequately addressed by architect's solutions
            if any(p for p in security_problems_identified) and not any(
                "security" in sol for sol in architect_solutions_proposed
            ):
                logger.info("Detected conflict type: SECURITY_VS_ARCHITECTURE")
                return "SECURITY_VS_ARCHITECTURE"

        # Check for FUNDAMENTAL_FLAW_DETECTION by Devils_Advocate
        if devils_advocate_output and isinstance(devils_advocate_output, dict):
            if devils_advocate_output.get("conflict_found", False):
                logger.info("Detected conflict type: FUNDAMENTAL_FLAW_DETECTION")
                return "FUNDAMENTAL_FLAW_DETECTION"

        logger.info("Detected conflict type: GENERAL_DISAGREEMENT")
        return "GENERAL_DISAGREEMENT"

    def resolve_conflict(
        self, debate_history: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Analyzes debate history and attempts to resolve conflicts or malformed outputs.
        This method is designed to be invoked when a persona's output is deemed problematic.

        Args:
            debate_history: A list of dictionaries, where each dictionary represents a turn
                            in the debate, containing "persona" and "output".

        Returns:
            An optional dictionary representing a resolved output, or None if no resolution
            could be found through automated means. The resolved output will include a
            "resolution_strategy" field.
        """
        logger.warning("ConflictResolutionManager: Attempting to resolve conflict...")

        if not debate_history:
            logger.warning(
                "ConflictResolutionManager: Debate history is empty. Cannot resolve."
            )
            return self._manual_intervention_fallback("Empty debate history.")

        latest_turn = debate_history[-1]
        latest_output = latest_turn.get("output")
        latest_persona_name = latest_turn.get("persona", "Unknown")

        # Check if the latest output is problematic (malformed_blocks or conflict_found)
        is_problematic = False
        if isinstance(latest_output, dict):
            if latest_output.get("malformed_blocks") or (
                latest_output.get("conflict_found") is True
            ):
                is_problematic = True
        elif isinstance(latest_output, str):
            try:
                json.loads(latest_output)
            except json.JSONDecodeError:
                is_problematic = True

        if is_problematic:
            logger.warning(
                f"ConflictResolutionManager: Detected problematic output from {latest_persona_name}. Attempting self-correction."
            )

            resolved_output = self._retry_persona_with_feedback(
                latest_persona_name, debate_history
            )
            if resolved_output:
                logger.info(
                    f"ConflictResolutionManager: Successfully self-corrected output from {latest_persona_name}."
                )
                return {
                    "resolution_strategy": "self_correction_retry",
                    "resolved_output": resolved_output,
                    "resolution_summary": f"Persona '{latest_persona_name}' self-corrected its output after receiving validation feedback.",
                    "malformed_blocks": [
                        {
                            "type": "SELF_CORRECTION_SUCCESS",
                            "message": "Persona self-corrected.",
                        }
                    ],
                }
            else:
                logger.warning(
                    f"ConflictResolutionManager: Self-correction failed for {latest_persona_name}. Falling back to synthesis/manual intervention."
                )

        # Strategy 1: Attempt to parse if the latest output is a string that looks like JSON
        if isinstance(latest_output, str):
            try:
                parsed_latest_output = json.loads(latest_output)
                logger.info(
                    f"ConflictResolutionManager: Successfully parsed string output from {latest_persona_name}."
                )
                return {
                    "resolution_strategy": "parsed_malformed_string",
                    "resolved_output": parsed_latest_output,
                    "resolution_summary": f"Successfully parsed malformed string output from {latest_persona_name}.",
                    "malformed_blocks": [
                        {
                            "type": "PARSED_STRING_OUTPUT",
                            "message": "Successfully parsed string output.",
                        }
                    ],
                }
            except json.JSONDecodeError:
                logger.debug(
                    f"ConflictResolutionManager: Latest output from {latest_persona_name} is a malformed string, cannot parse directly."
                )

        # NEW: Conflict resolution strategy based on type
        conflict_type = self._detect_conflict_type(debate_history)
        logger.info(f"Detected conflict type: {conflict_type}")

        # Extract the latest ConflictReport if available, especially from Devils_Advocate
        latest_conflict_report = None
        for turn in reversed(debate_history):
            if turn["persona"] == "Devils_Advocate" and isinstance(turn["output"], dict):
                try:
                    latest_conflict_report = ConflictReport.model_validate(turn["output"])
                    break
                except ValidationError:
                    continue

        # --- START NEW LOGIC FOR MISSING CODEBASE CONTEXT ---
        if latest_conflict_report and (
            "lack of information" in latest_conflict_report.summary.lower()
            or "no codebase context" in latest_conflict_report.summary.lower()
        ):
            logger.warning("ConflictResolutionManager: Detected conflict due to missing codebase context.")
            # Construct a SelfImprovementAnalysisOutputV1 compliant response
            resolved_output_data = SelfImprovementAnalysisOutputV1(
                ANALYSIS_SUMMARY="Cannot perform analysis without codebase access. Please provide access to the Project Chimera codebase.",
                IMPACTFUL_SUGGESTIONS=[
                    SuggestionItem(
                        AREA="Maintainability",
                        PROBLEM="Critical lack of codebase access prevents meaningful code-level analysis and improvements. The system cannot perform security, robustness, or detailed maintainability analyses without the codebase.",
                        PROPOSED_SOLUTION="Establish a mechanism for providing the Project Chimera codebase and its context. This includes providing relevant files, their purpose, architecture, and any prior analysis. A `docs/project_chimera_context.md` file is proposed to guide this collection.",
                        EXPECTED_IMPACT="Enables the self-improvement process to proceed effectively, allowing for specific vulnerability identification, optimization opportunities, and actionable code modifications. This directly addresses the 'Maintainability' aspect of the self-improvement goals.",
                        CODE_CHANGES_SUGGESTED=[
                            {
                                "FILE_PATH": "docs/project_chimera_context.md",
                                "ACTION": "CREATE",
                                "FULL_CONTENT": "# Project Chimera Codebase Context\n\nThis document outlines the codebase structure and key files required for AI analysis. Please populate this file with relevant information."
                            }
                        ]
                    )
                ],
                malformed_blocks=[
                    {
                        "type": "CODEBASE_ACCESS_REQUIRED",
                        "message": "Analysis requires codebase access.",
                    }
                ],
            ).model_dump(by_alias=True) # Ensure it's a dict

            return {
                "resolution_strategy": "missing_codebase_context",
                "resolved_output": resolved_output_data,
                "resolution_summary": "Cannot perform analysis without codebase access. Please provide access to the Project Chimera codebase.",
                "malformed_blocks": [
                    {"type": "CODEBASE_ACCESS_REQUIRED", "message": "Analysis requires codebase access."}
                ],
            }
        # --- END NEW LOGIC FOR MISSING CODEBASE CONTEXT ---


        if conflict_type == "SECURITY_VS_ARCHITECTURE":
            logger.info(
                "Applying specific resolution for SECURITY_VS_ARCHITECTURE conflict."
            )
            # Example logic: Prioritize security concerns, ask for architectural solutions that address them
            resolution_summary = "Security concerns were identified by Security_Auditor but not adequately addressed by Code_Architect. Prioritizing security in the resolution."
            # This would typically involve another LLM call to synthesize a solution that balances both.
            # For now, we'll synthesize from history, but with a specific summary.
            synthesized_output = self._synthesize_from_history(
                latest_output, debate_history[:-1], resolution_summary
            )
            if synthesized_output:
                return {
                    "resolution_strategy": "security_vs_architecture_synthesis",
                    "resolved_output": synthesized_output,
                    "resolution_summary": resolution_summary,
                    "malformed_blocks": [
                        {
                            "type": "SECURITY_VS_ARCHITECTURE_RESOLVED",
                            "message": resolution_summary,
                        }
                    ],
                }
            else:
                logger.warning(
                    "SECURITY_VS_ARCHITECTURE synthesis failed. Falling back to general resolution."
                )

        elif conflict_type == "FUNDAMENTAL_FLAW_DETECTION":
            logger.info(
                "Applying specific resolution for FUNDAMENTAL_FLAW_DETECTION conflict."
            )
            # Example logic: If Devils_Advocate found a fundamental flaw, it needs to be addressed directly.
            resolution_summary = "Devils_Advocate identified a fundamental flaw. The resolution focuses on addressing this core issue."
            # This might involve re-prompting a persona to fix the flaw, or a synthesis that explicitly incorporates the flaw.
            synthesized_output = self._synthesize_from_history(
                latest_output, debate_history[:-1], resolution_summary
            )
            if synthesized_output:
                return {
                    "resolution_strategy": "fundamental_flaw_resolution_synthesis",
                    "resolved_output": synthesized_output,
                    "resolution_summary": resolution_summary,
                    "malformed_blocks": [
                        {
                            "type": "FUNDAMENTAL_FLAW_RESOLVED",
                            "message": resolution_summary,
                        }
                    ],
                }
            else:
                logger.warning(
                    "FUNDAMENTAL_FLAW_DETECTION synthesis failed. Falling back to general resolution."
                )
        else:
            logger.info("Applying general conflict resolution strategy.")
            # Strategy 2: Synthesize from previous valid turns (General Disagreement)
            valid_turns = [
                turn
                for turn in debate_history[:-1]
                if isinstance(turn.get("output"), dict)
                and not turn["output"].get("malformed_blocks")
            ]

            if len(valid_turns) >= 2:
                logger.info(
                    f"ConflictResolutionManager: Found {len(valid_turns)} previous valid turns. Attempting synthesis."
                )
                synthesized_output = self._synthesize_from_history(
                    latest_output, valid_turns
                )
                if synthesized_output:
                    logger.info(
                        "ConflictResolutionManager: Successfully synthesized from history."
                    )
                    return {
                        "resolution_strategy": "synthesis_from_history",
                        "resolved_output": synthesized_output,
                        "resolution_summary": "Synthesized a coherent output from previous valid debate turns.",
                        "malformed_blocks": [
                            {
                                "type": "SYNTHESIS_FROM_HISTORY",
                                "message": "Automated synthesis from history due to problematic output.",
                            }
                        ],
                    }
                else:
                    logger.warning(
                        "ConflictResolutionManager: Synthesis from history failed."
                    )

        # Strategy 3: Fallback to a generic placeholder or flag for manual intervention
        logger.warning(
            "ConflictResolutionManager: Automated resolution strategies exhausted."
        )
        return self._manual_intervention_fallback(
            f"Automated resolution failed for output from {latest_persona_name}. Latest output snippet: {str(latest_output)[:200]}..."
        )

    def _synthesize_from_history(
        self,
        problematic_output: Any,
        valid_turns: List[Dict[str, Any]],
        custom_summary: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Attempts to synthesize a coherent output from previous valid turns.
        """
        if valid_turns:
            last_valid_output = valid_turns[-1]["output"]
            resolution_summary = (
                custom_summary
                if custom_summary
                else f"Conflict detected after {valid_turns[-1]['persona']}'s turn. Automated synthesis from previous turns was attempted. Original problematic output: {str(problematic_output)[:100]}..."
            )

            if isinstance(last_valid_output, dict):
                synthesized = last_valid_output.copy()
                synthesized["CONFLICT_RESOLUTION_ATTEMPT"] = resolution_summary
                synthesized.setdefault("malformed_blocks", []).append(
                    {
                        "type": "SYNTHESIS_ATTEMPT",
                        "message": "Automated synthesis from history due to problematic output.",
                    }
                )
                return synthesized
            elif isinstance(last_valid_output, str):
                return {
                    "general_output": f"{last_valid_output}\n\n[Automated Conflict Resolution Note: {resolution_summary}]",
                    "malformed_blocks": [
                        {
                            "type": "SYNTHESIS_ATTEMPT",
                            "message": "Automated synthesis from history due to problematic output.",
                        }
                    ],
                }

        return None

    def _manual_intervention_fallback(self, message: str) -> Dict[str, Any]:
        """
        Returns a structured output indicating that manual intervention is required.
        """
        logger.error(
            f"ConflictResolutionManager: Manual intervention required: {message}"
        )
        return {
            "resolution_strategy": "manual_intervention",
            "resolved_output": None,
            "resolution_summary": f"Automated conflict resolution failed. Manual review required: {message}",
            "malformed_blocks": [
                {"type": "MANUAL_INTERVENTION_REQUIRED", "message": message}
            ],
        }

    def _retry_persona_with_feedback(
        self, persona_name: str, debate_history: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Re-invokes the persona with explicit feedback on its previous problematic output.
        """
        if (
            not self.llm_provider
            or not self.llm_provider.tokenizer
            or not self.persona_manager
        ):
            logger.error(
                "LLMProvider, Tokenizer, or PersonaManager not set in ConflictResolutionManager. Cannot self-correct."
            )
            return None

        initial_prompt_from_history = (
            "Original task context not explicitly captured in history."
        )
        if (
            debate_history
            and debate_history[0].get("output")
            and isinstance(debate_history[0]["output"], dict)
        ):
            initial_prompt_from_history = debate_history[0]["output"].get(
                "initial_prompt", initial_prompt_from_history
            )
        if (
            initial_prompt_from_history
            == "Original task context not explicitly captured in history."
        ):
            initial_prompt_from_history = f"The debate started with the goal of: {debate_history[0].get('initial_prompt', 'analyzing a problem.')}"

        persona_config = self.persona_manager.get_adjusted_persona_config(persona_name)
        if not persona_config:
            logger.error(
                f"Persona config not found for {persona_name}. Cannot self-correct."
            )
            return None

        problematic_turn = debate_history[-1]
        problematic_output = problematic_turn.get("output")

        error_message = "Previous output was malformed or did not adhere to the schema."
        if isinstance(problematic_output, dict) and problematic_output.get(
            "malformed_blocks"
        ):
            validation_error_block = next(
                (
                    b
                    for b in problematic_output["malformed_blocks"]
                    if b.get("type") == "SCHEMA_VALIDATION_ERROR"
                ),
                None,
            )
            if validation_error_block:
                error_message = f"Specific validation error: {validation_error_block.get('message', 'Unknown schema error.')}"
            else:
                error_message = f"Specific validation errors: {json.dumps(problematic_output['malformed_blocks'], indent=2)}"
        elif isinstance(problematic_output, str):
            error_message = f"Previous output was a malformed string: '{self.output_parser._clean_llm_output(str(problematic_output))[:200]}...'"

        feedback_prompt = f"""
        Your previous response for the persona '{persona_name}' was problematic.

        Original Request Context:
        {initial_prompt_from_history}

        Your Previous Output (which failed validation):
        ```
        {self.output_parser._clean_llm_output(str(problematic_output))[:1000]}
        ```

        CRITICAL ERROR FEEDBACK:
        {error_message}

        You MUST correct this. Ensure your new output is a SINGLE, VALID JSON OBJECT, strictly adhering to the schema provided in your system prompt.
        DO NOT include any conversational text or markdown fences outside the JSON. Focus solely on providing a correct, valid response.
        """

        output_schema_class = self.persona_manager.PERSONA_OUTPUT_SCHEMAS.get(
            persona_name.replace("_TRUNCATED", ""), GeneralOutput
        )

        full_system_prompt_parts = [persona_config.system_prompt]
        full_system_prompt_parts.append(SHARED_JSON_INSTRUCTIONS)
        full_system_prompt_parts.append(
            f"**JSON Schema for {output_schema_class.__name__}:**\n```json\n{json.dumps(output_schema_class.model_json_schema(), indent=2)}\n```"
        )
        final_system_prompt = "\n\n".join(full_system_prompt_parts)

        for retry_attempt in range(self.max_self_correction_retries):
            logger.info(
                f"Attempting self-correction for {persona_name} (Retry {retry_attempt + 1}/{self.max_self_correction_retries})."
            )
            try:
                if (
                    not self.llm_provider
                    or not self.llm_provider.tokenizer
                    or not self.persona_manager
                ):
                    raise ChimeraError(
                        "LLM Provider, Tokenizer, or PersonaManager not available for self-correction."
                    )

                actual_model_max_output_tokens = (
                    self.llm_provider.tokenizer.max_output_tokens
                )
                effective_max_output_tokens = min(
                    persona_config.max_tokens, actual_model_max_output_tokens
                )

                raw_llm_output, input_tokens, output_tokens, is_truncated = (
                    self.llm_provider.generate(
                        prompt=feedback_prompt,
                        system_prompt=final_system_prompt,
                        temperature=persona_config.temperature,
                        max_tokens=effective_max_output_tokens,
                        persona_config=persona_config,
                        requested_model_name=self.llm_provider.model_name,
                        output_schema=output_schema_class,
                    )
                )

                corrected_output = self.output_parser.parse_and_validate(
                    raw_llm_output, output_schema_class
                )

                if not corrected_output.get("malformed_blocks"):
                    logger.info(
                        f"Self-correction successful for {persona_name} on retry {retry_attempt + 1}."
                    )
                    return corrected_output
                else:
                    logger.warning(
                        f"Self-correction attempt {retry_attempt + 1} for {persona_name} still resulted in malformed blocks: {corrected_output.get('malformed_blocks')}"
                    )
                    feedback_prompt += f"\n\nPrevious self-correction attempt also failed. New errors: {json.dumps(corrected_output['malformed_blocks'], indent=2)}\nCRITICAL: You MUST fix these new errors."

            except Exception as e:
                logger.error(
                    f"Error during self-correction retry for {persona_name}: {e}",
                    exc_info=True,
                )
                feedback_prompt += f"\n\nPrevious self-correction attempt failed with an internal error: {str(e)}\nCRITICAL: Address this error."

        logger.warning(
            f"Self-correction failed for {persona_name} after {self.max_self_correction_retries} retries."
        )
        return None