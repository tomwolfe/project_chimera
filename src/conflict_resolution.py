import json
import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING

# NEW: Import necessary classes for self-correction
from src.llm_provider import GeminiProvider
from src.models import PersonaConfig, GeneralOutput, LLMOutput, CritiqueOutput, ConflictReport, SelfImprovementAnalysisOutputV1, ContextAnalysisOutput, ConfigurationAnalysisOutput, DeploymentAnalysisOutput
from src.utils.output_parser import LLMOutputParser
from src.llm_tokenizers.gemini_tokenizer import GeminiTokenizer # MODIFIED: Updated import path
from src.config.settings import ChimeraSettings
from src.constants import SHARED_JSON_INSTRUCTIONS
from src.exceptions import ChimeraError # Import ChimeraError for raising

# Use TYPE_CHECKING to avoid circular import at runtime
if TYPE_CHECKING:
    from src.persona_manager import PersonaManager

logger = logging.getLogger(__name__)

class ConflictResolutionManager:
    """
    Manages the resolution of conflicts or malformed outputs from LLM personas.
    Implements strategies to attempt to recover or synthesize a coherent output.
    """

    def __init__(self, llm_provider: Optional[GeminiProvider] = None, persona_manager: Optional["PersonaManager"] = None):
        logger.info("ConflictResolutionManager initialized.")
        self.llm_provider = llm_provider
        self.persona_manager = persona_manager
        self.max_self_correction_retries = 2 # Max retries for self-correction
        self.settings = ChimeraSettings() # Initialize settings
        self.output_parser = LLMOutputParser() # Initialize parser here

    def resolve_conflict(self, debate_history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
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
            logger.warning("ConflictResolutionManager: Debate history is empty. Cannot resolve.")
            return self._manual_intervention_fallback("Empty debate history.")

        latest_turn = debate_history[-1]
        latest_output = latest_turn.get('output')
        latest_persona_name = latest_turn.get('persona', 'Unknown')

        # NEW: Check if the latest output is problematic (malformed_blocks or conflict_found)
        is_problematic = False
        if isinstance(latest_output, dict):
            if latest_output.get('malformed_blocks') or (latest_output.get('conflict_found') is True):
                is_problematic = True
        elif isinstance(latest_output, str):
            # If it's a string, it's problematic unless it can be parsed as valid JSON
            try:
                json.loads(latest_output)
            except json.JSONDecodeError:
                is_problematic = True

        if is_problematic:
            logger.warning(f"ConflictResolutionManager: Detected problematic output from {latest_persona_name}. Attempting self-correction.")
            
            # Attempt self-correction by re-invoking the persona with feedback
            resolved_output = self._retry_persona_with_feedback(latest_persona_name, debate_history)
            if resolved_output:
                logger.info(f"ConflictResolutionManager: Successfully self-corrected output from {latest_persona_name}.")
                return {
                    "resolution_strategy": "self_correction_retry",
                    "resolved_output": resolved_output,
                    "resolution_summary": f"Persona '{latest_persona_name}' self-corrected its output after receiving validation feedback.",
                    "malformed_blocks": [{"type": "SELF_CORRECTION_SUCCESS", "message": "Persona self-corrected."}]
                }
            else:
                logger.warning(f"ConflictResolutionManager: Self-correction failed for {latest_persona_name}. Falling back to synthesis/manual intervention.")
        
        # Strategy 1: Attempt to parse if the latest output is a string that looks like JSON
        if isinstance(latest_output, str):
            try:
                parsed_latest_output = json.loads(latest_output) # Assuming this is a valid JSON string
                logger.info(f"ConflictResolutionManager: Successfully parsed string output from {latest_persona_name}.")
                return {
                    "resolution_strategy": "parsed_malformed_string",
                    "resolved_output": parsed_latest_output,
                    "resolution_summary": f"Successfully parsed malformed string output from {latest_persona_name}.",
                    "malformed_blocks": [{"type": "PARSED_STRING_OUTPUT", "message": "Successfully parsed string output."}]
                }
            except json.JSONDecodeError:
                logger.debug(f"ConflictResolutionManager: Latest output from {latest_persona_name} is a malformed string, cannot parse directly.")
                # If parsing fails, fall through to the next strategy

        # Strategy 2: Synthesize from previous valid turns
        # Look for at least two previous valid, structured outputs to synthesize from
        valid_turns = [
            turn for turn in debate_history[:-1] # Exclude the latest problematic turn
            if isinstance(turn.get('output'), dict) and not turn['output'].get('malformed_blocks')
        ]

        if len(valid_turns) >= 2:
            logger.info(f"ConflictResolutionManager: Found {len(valid_turns)} previous valid turns. Attempting synthesis.")
            synthesized_output = self._synthesize_from_history(latest_output, valid_turns)
            if synthesized_output:
                logger.info("ConflictResolutionManager: Successfully synthesized from history.")
                return {
                    "resolution_strategy": "synthesis_from_history",
                    "resolved_output": synthesized_output,
                    "resolution_summary": "Synthesized a coherent output from previous valid debate turns.",
                    "malformed_blocks": [{"type": "SYNTHESIS_FROM_HISTORY", "message": "Automated synthesis from history due to problematic output."}]
                }
            else:
                logger.warning("ConflictResolutionManager: Synthesis from history failed.")

        # Strategy 3: Fallback to a generic placeholder or flag for manual intervention
        logger.warning("ConflictResolutionManager: Automated resolution strategies exhausted.")
        return self._manual_intervention_fallback(
            f"Automated resolution failed for output from {latest_persona_name}. Latest output snippet: {str(latest_output)[:200]}..."
        )

    def _synthesize_from_history(self, problematic_output: Any, valid_turns: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        (Placeholder) Attempts to synthesize a coherent output from previous valid turns.
        In a real implementation, this would involve another LLM call or a sophisticated
        rule-based system to combine insights.
        """
        # For now, a simple heuristic: try to combine summaries or take the most recent valid one.
        # This could be an LLM call to an "Impartial_Arbitrator" or "General_Synthesizer" persona.
        
        # Simple approach: take the last valid output and append a note about the conflict.
        if valid_turns:
            last_valid_output = valid_turns[-1]['output']
            resolution_summary = f"Conflict detected after {valid_turns[-1]['persona']}'s turn. Automated synthesis from previous turns was attempted. Original problematic output: {str(problematic_output)[:100]}..."
            
            # Try to merge or append to the last valid output
            if isinstance(last_valid_output, dict):
                synthesized = last_valid_output.copy()
                synthesized['CONFLICT_RESOLUTION_ATTEMPT'] = resolution_summary
                synthesized.setdefault('malformed_blocks', []).append({
                    "type": "SYNTHESIS_ATTEMPT",
                    "message": "Automated synthesis from history due to problematic output."
                })
                return synthesized
            elif isinstance(last_valid_output, str):
                return {
                    "general_output": f"{last_valid_output}\n\n[Automated Conflict Resolution Note: {resolution_summary}]",
                    "malformed_blocks": [{"type": "SYNTHESIS_ATTEMPT", "message": "Automated synthesis from history due to problematic output."}]
                }
        
        return None

    def _manual_intervention_fallback(self, message: str) -> Dict[str, Any]:
        """
        Returns a structured output indicating that manual intervention is required.
        """
        logger.error(f"ConflictResolutionManager: Manual intervention required: {message}")
        return {
            "resolution_strategy": "manual_intervention",
            "resolved_output": None,
            "resolution_summary": f"Automated conflict resolution failed. Manual review required: {message}",
            "malformed_blocks": [{"type": "MANUAL_INTERVENTION_REQUIRED", "message": message}]
        }

    def _retry_persona_with_feedback(self, persona_name: str, debate_history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Re-invokes the persona with explicit feedback on its previous problematic output.
        """
        if not self.llm_provider or not self.llm_provider.tokenizer or not self.persona_manager:
            logger.error("LLMProvider, Tokenizer, or PersonaManager not set in ConflictResolutionManager. Cannot self-correct.")
            return None

        # The initial prompt is typically the first entry in the debate history
        # or can be passed explicitly to the SocraticDebate constructor.
        # For this context, we'll assume the initial prompt is available from the SocraticDebate instance
        # that initialized this ConflictResolutionManager.
        # Since this method is called from core.py, core.py should provide the initial prompt.
        # For now, we'll try to extract it from the first turn if available.
        initial_prompt_from_history = "Original task context not explicitly captured in history."
        if debate_history and debate_history[0].get('output') and isinstance(debate_history[0]['output'], dict):
            # Assuming the first turn's output might contain the initial prompt if it's a ContextAnalysisOutput or similar
            initial_prompt_from_history = debate_history[0]['output'].get('initial_prompt', initial_prompt_from_history)
        # Fallback to a more general summary of the debate if initial_prompt is still not found
        if initial_prompt_from_history == "Original task context not explicitly captured in history.":
            initial_prompt_from_history = f"The debate started with the goal of: {debate_history[0].get('initial_prompt', 'analyzing a problem.')}"


        # Get the persona's original system prompt and config
        persona_config = self.persona_manager.get_adjusted_persona_config(persona_name)
        if not persona_config:
            logger.error(f"Persona config not found for {persona_name}. Cannot self-correct.")
            return None

        # Get the problematic output and its error details
        problematic_turn = debate_history[-1]
        problematic_output = problematic_turn.get('output')
        
        error_message = "Previous output was malformed or did not adhere to the schema."
        if isinstance(problematic_output, dict) and problematic_output.get('malformed_blocks'):
            # NEW: Make feedback more direct by extracting the core error message.
            validation_error_block = next((b for b in problematic_output['malformed_blocks'] if b.get('type') == 'SCHEMA_VALIDATION_ERROR'), None)
            if validation_error_block:
                error_message = f"Specific validation error: {validation_error_block.get('message', 'Unknown schema error.')}"
            else:
                error_message = f"Specific validation errors: {json.dumps(problematic_output['malformed_blocks'], indent=2)}"
        elif isinstance(problematic_output, str):
            error_message = f"Previous output was a malformed string: '{self.output_parser._clean_llm_output(problematic_output)[:200]}...'" # MODIFIED: Clean string output
        
        # Construct the feedback prompt
        feedback_prompt = f"""
        Your previous response for the persona '{persona_name}' was problematic.
        
        Original Request Context:
        {initial_prompt_from_history}

        Your Previous Output (which failed validation):
        ```
        {self.output_parser._clean_llm_output(str(problematic_output))[:1000]} # MODIFIED: Use cleaned output here
        ```

        CRITICAL ERROR FEEDBACK:
        {error_message}

        You MUST correct this. Ensure your new output is a SINGLE, VALID JSON OBJECT, strictly adhering to the schema provided in your system prompt.
        DO NOT include any conversational text or markdown fences outside the JSON. Focus solely on providing a correct, valid response.
        """
        
        # Get the schema for the persona's output
        # This requires persona_manager to have access to PERSONA_OUTPUT_SCHEMAS
        output_schema_class = self.persona_manager.PERSONA_OUTPUT_SCHEMAS.get(persona_name.replace("_TRUNCATED", ""), GeneralOutput)
        
        full_system_prompt_parts = [persona_config.system_prompt]
        full_system_prompt_parts.append(SHARED_JSON_INSTRUCTIONS)
        full_system_prompt_parts.append(f"**JSON Schema for {output_schema_class.__name__}:**\n```json\n{json.dumps(output_schema_class.model_json_schema(), indent=2)}\n```")
        final_system_prompt = "\n\n".join(full_system_prompt_parts)

        # Re-invoke the LLM for self-correction
        for retry_attempt in range(self.max_self_correction_retries):
            logger.info(f"Attempting self-correction for {persona_name} (Retry {retry_attempt + 1}/{self.max_self_correction_retries}).")
            try:
                # Ensure LLMProvider and Tokenizer are available
                if not self.llm_provider or not self.llm_provider.tokenizer or not self.persona_manager:
                    raise ChimeraError("LLM Provider, Tokenizer, or PersonaManager not available for self-correction.")

                actual_model_max_output_tokens = self.llm_provider.tokenizer.max_output_tokens
                effective_max_output_tokens = min(persona_config.max_tokens, actual_model_max_output_tokens)

                raw_llm_output, input_tokens, output_tokens, is_truncated = self.llm_provider.generate(
                    prompt=feedback_prompt,
                    system_prompt=final_system_prompt,
                    temperature=persona_config.temperature,
                    max_tokens=effective_max_output_tokens, # Use the calculated effective max tokens
                    persona_config=persona_config,
                    requested_model_name=self.llm_provider.model_name, # Use the provider's current model
                    output_schema=output_schema_class # NEW: Pass the schema for early validation
                )
                
                # Attempt to parse and validate the corrected output
                # This call will now benefit from the early schema validation in llm_provider.generate
                corrected_output = self.output_parser.parse_and_validate(raw_llm_output, output_schema_class)
                
                if not corrected_output.get('malformed_blocks'):
                    logger.info(f"Self-correction successful for {persona_name} on retry {retry_attempt + 1}.")
                    return corrected_output
                else:
                    logger.warning(f"Self-correction attempt {retry_attempt + 1} for {persona_name} still resulted in malformed blocks: {corrected_output.get('malformed_blocks')}")
                    # Append new errors to the feedback prompt for the next retry
                    feedback_prompt += f"\n\nPrevious self-correction attempt also failed. New errors: {json.dumps(corrected_output['malformed_blocks'], indent=2)}\nCRITICAL: You MUST fix these new errors."

            except Exception as e:
                logger.error(f"Error during self-correction retry for {persona_name}: {e}", exc_info=True)
                feedback_prompt += f"\n\nPrevious self-correction attempt failed with an internal error: {str(e)}\nCRITICAL: Address this error."

        logger.warning(f"Self-correction failed for {persona_name} after {self.max_self_correction_retries} retries.")
        return None