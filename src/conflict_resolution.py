# src/conflict_resolution.py
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConflictResolutionManager:
    """
    Manages the resolution of conflicts or malformed outputs from LLM personas.
    Implements strategies to attempt to recover or synthesize a coherent output.
    """

    def __init__(self):
        logger.info("ConflictResolutionManager initialized.")

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
        latest_persona = latest_turn.get('persona', 'Unknown')

        # Strategy 1: Attempt to parse if the latest output is a string that looks like JSON
        if isinstance(latest_output, str):
            try:
                # Attempt to parse the string as JSON
                parsed_latest_output = json.loads(latest_output)
                logger.info(f"ConflictResolutionManager: Successfully parsed string output from {latest_persona}.")
                return {
                    "resolution_strategy": "parsed_malformed_string",
                    "resolved_output": parsed_latest_output,
                    "resolution_summary": f"Successfully parsed malformed string output from {latest_persona}.",
                    "malformed_blocks": [{"type": "PARSED_STRING_OUTPUT", "message": "Successfully parsed string output."}]
                }
            except json.JSONDecodeError:
                logger.debug(f"ConflictResolutionManager: Latest output from {latest_persona} is a malformed string, cannot parse directly.")
                # If parsing fails, fall through to the next strategy

        # Strategy 2: Synthesize from previous valid turns
        # Look for at least two previous valid, structured outputs to synthesize from
        valid_turns = [
            turn for turn in debate_history[:-1] # Exclude the latest problematic turn
            if isinstance(turn.get('output'), dict) and not turn['output'].get('malformed_blocks')
        ]

        if len(valid_turns) >= 2:
            logger.info(f"ConflictResolutionManager: Found {len(valid_turns)} previous valid turns. Attempting synthesis.")
            synthesis_result = self._synthesize_from_history(latest_output, valid_turns)
            if synthesis_result:
                logger.info("ConflictResolutionManager: Successfully synthesized from history.")
                return {
                    "resolution_strategy": "synthesis_from_history",
                    "resolved_output": synthesis_result,
                    "resolution_summary": "Synthesized a coherent output from previous valid debate turns.",
                    "malformed_blocks": [{"type": "SYNTHESIS_FROM_HISTORY", "message": "Synthesized from previous valid turns."}]
                }
            else:
                logger.warning("ConflictResolutionManager: Synthesis from history failed.")

        # Strategy 3: Fallback to a generic placeholder or flag for manual intervention
        logger.warning("ConflictResolutionManager: Automated resolution strategies exhausted.")
        return self._manual_intervention_fallback(
            f"Automated resolution failed for output from {latest_persona}. Latest output snippet: {str(latest_output)[:200]}..."
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