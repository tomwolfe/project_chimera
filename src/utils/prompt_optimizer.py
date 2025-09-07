# src/utils/prompt_optimizer.py
import logging
from typing import Dict, Any, List, Optional
from src.tokenizers import Tokenizer
from src.config.settings import ChimeraSettings # Import ChimeraSettings
import re # Added for prompt section parsing

logger = logging.getLogger(__name__)

class PromptOptimizer:
    """Optimizes prompts for various personas based on context and token limits."""

    def __init__(self, tokenizer: Tokenizer, settings: ChimeraSettings):
        self.tokenizer = tokenizer
        self.settings = settings

    def optimize_prompt(
        self,
        prompt: str,
        persona_name: str,
        max_output_tokens_for_turn: int,
    ) -> str:
        """
        Optimizes a prompt for a specific persona based on actual token usage and persona-specific limits.
        This method aims to reduce the input prompt size if it, combined with the expected output,
        exceeds a reasonable threshold for the persona, or if the overall token budget is constrained.
        """
        # Calculate current prompt tokens
        prompt_tokens = self.tokenizer.count_tokens(prompt)
        
        # Get persona-specific token limits from settings
        # Use a default if not explicitly defined for the persona
        # Note: max_output_tokens_for_turn is already passed, so we use that as the target output.
        # The persona's max_tokens from PersonaConfig is the hard limit for output.
        # Here, we're concerned with the *input* prompt's size.
        persona_input_token_limit = self.settings.max_tokens_per_persona.get(
            persona_name, self.settings.default_max_input_tokens_per_persona
        )

        # Heuristic: If the current prompt tokens are already very high,
        # or if the combined input + expected output tokens exceed a certain percentage
        # of the persona's total capacity (e.g., 80% of its max_tokens from PersonaConfig),
        # then we should aggressively optimize the input.
        # The `max_output_tokens_for_turn` is the *actual* budget for the output,
        # so we should consider it.
        
        # Calculate a soft limit for the input prompt based on the persona's overall capacity
        # and the expected output size.
        # A persona's total capacity is its max_tokens (from PersonaConfig, which is passed to _execute_llm_turn)
        # We want to ensure input + output fits within that.
        # Let's assume `max_output_tokens_for_turn` is the effective max output.
        # So, the input should ideally be `persona_total_capacity - max_output_tokens_for_turn`.
        
        # Prioritize `persona_input_token_limit` from settings for input control.
        effective_input_limit = persona_input_token_limit

        # --- Enhanced logic for Self_Improvement_Analyst ---
        if persona_name == "Self_Improvement_Analyst":
            # Apply a more aggressive base limit for this verbose persona
            effective_input_limit = min(effective_input_limit, 2500) # Example: Cap input at 2500 tokens
            
            # If the prompt is still too long, try to intelligently summarize/prioritize sections
            if prompt_tokens > effective_input_limit:
                logger.debug(f"Self_Improvement_Analyst prompt exceeds aggressive input limit ({prompt_tokens}/{effective_input_limit}). Attempting intelligent summarization.")
                
                # This is a conceptual implementation. A real one would involve:
                # 1. Identifying distinct sections in the Self_Improvement_Analyst prompt (e.g., "SECURITY ANALYSIS:", "TOKEN OPTIMIZATION:").
                # 2. Prioritizing sections based on keywords in the current debate context or intermediate results.
                # 3. Summarizing less critical sections or removing them entirely if budget is very tight.
                
                # For now, a more sophisticated truncation that tries to keep critical instructions.
                # This example attempts to preserve the CRITICAL INSTRUCTION blocks.
                optimized_prompt = self._optimize_self_improvement_analyst_prompt(
                    prompt, effective_input_limit
                )
                
                if self.tokenizer.count_tokens(optimized_prompt) < prompt_tokens:
                    logger.info(
                        f"Prompt for {persona_name} truncated from {prompt_tokens} to {self.tokenizer.count_tokens(optimized_prompt)} tokens."
                    )
                    return optimized_prompt
                else:
                    logger.warning("Intelligent optimization for Self_Improvement_Analyst did not significantly reduce tokens. Falling back to general truncation.")
        # --- End enhanced logic ---

        if prompt_tokens > effective_input_limit:
            logger.warning(
                f"{persona_name} prompt exceeds effective input token limit ({prompt_tokens}/{effective_input_limit}). Optimizing..."
            )
            
            optimized_prompt = self.tokenizer.truncate_to_token_limit(
                prompt, effective_input_limit, 
                truncation_indicator="\n\n[TRUNCATED - focusing on most critical aspects]"
            )
            
            logger.info(
                f"Prompt for {persona_name} truncated from {prompt_tokens} to {self.tokenizer.count_tokens(optimized_prompt)} tokens."
            )
            return optimized_prompt
        
        return prompt

    def _optimize_self_improvement_analyst_prompt(self, prompt: str, max_tokens: int) -> str:
        """
        Intelligently optimizes the Self_Improvement_Analyst prompt by prioritizing critical sections.
        This is a heuristic-based approach.
        """
        sections = {
            "core_mission": "",
            "critical_instruction_absolute_adherence": "",
            "security_analysis": "",
            "token_optimization": "",
            "testing_strategy": "",
            "ai_reasoning_quality": "",
            "critical_json_output_instructions": "",
            "json_schema": "",
        }

        # Regex to extract sections based on headings/markers
        # More robust regex patterns to capture sections based on common markdown headings or explicit markers
        core_mission_match = re.search(r"(You are Project Chimera's Self-Improvement Analyst.*?)(?=\n---|\Z)", prompt, re.DOTALL)
        if core_mission_match:
            sections["core_mission"] = core_mission_match.group(1).strip()

        critical_instruction_match = re.search(r"(---\n\s*\*\*CRITICAL INSTRUCTION: ABSOLUTE ADHERENCE TO CONFLICT RESOLUTION\*\*.*?)(?=\n---|\Z)", prompt, re.DOTALL)
        if critical_instruction_match:
            sections["critical_instruction_absolute_adherence"] = critical_instruction_match.group(1).strip()

        security_analysis_match = re.search(r"(\*\*SECURITY ANALYSIS:\*\*.*?)(?=\*\*TOKEN OPTIMIZATION \(AI Efficiency\):\*\*|\*\*TESTING STRATEGY \(AI Robustness\):\*\*|\*\*AI REASONING QUALITY & DEBATE PROCESS IMPROVEMENT:\*\*|\n---|\Z)", prompt, re.DOTALL)
        if security_analysis_match:
            sections["security_analysis"] = security_analysis_match.group(1).strip()

        token_optimization_match = re.search(r"(\*\*TOKEN OPTIMIZATION \(AI Efficiency\):\*\*.*?)(?=\*\*TESTING STRATEGY \(AI Robustness\):\*\*|\*\*AI REASONING QUALITY & DEBATE PROCESS IMPROVEMENT:\*\*|\n---|\Z)", prompt, re.DOTALL)
        if token_optimization_match:
            sections["token_optimization"] = token_optimization_match.group(1).strip()
        
        testing_strategy_match = re.search(r"(\*\*TESTING STRATEGY \(AI Robustness\):\*\*.*?)(?=\*\*AI REASONING QUALITY & DEBATE PROCESS IMPROVEMENT:\*\*|\n---|\Z)", prompt, re.DOTALL)
        if testing_strategy_match:
            sections["testing_strategy"] = testing_strategy_match.group(1).strip()

        ai_reasoning_match = re.search(r"(\*\*AI REASONING QUALITY & DEBATE PROCESS IMPROVEMENT:\*\*.*?)(?=\n---|\Z)", prompt, re.DOTALL)
        if ai_reasoning_match:
            sections["ai_reasoning_quality"] = ai_reasoning_match.group(1).strip()

        json_instructions_match = re.search(r"(---\n\s*\*\*CRITICAL JSON OUTPUT INSTRUCTIONS: ABSOLUTELY MUST BE FOLLOWED\. STRICTLY ADHERE TO THE SCHEMA AND CODE CHANGE GUIDELINES\*\*.*?)(?=\*\*JSON Schema for SelfImprovementAnalysisOutput \(V1 data structure\):\*\*|\Z)", prompt, re.DOTALL)
        if json_instructions_match:
            sections["critical_json_output_instructions"] = json_instructions_match.group(1).strip()

        json_schema_match = re.search(r"(\*\*JSON Schema for SelfImprovementAnalysisOutput \(V1 data structure\):\*\*.*?)(?=\*\*Synthesize the following feedback into the specified JSON format:\*\*|\Z)", prompt, re.DOTALL)
        if json_schema_match:
            sections["json_schema"] = json_schema_match.group(1).strip()

        # Prioritize sections: Core mission, JSON instructions/schema, then specific analysis areas.
        # The order is crucial for effective truncation.
        # Core mission and JSON schema/instructions are always critical.
        # Specific analysis areas (security, token, testing, reasoning) can be dynamically prioritized
        # or truncated more aggressively if the overall prompt is too long.
        prioritized_sections = [
            sections["core_mission"],
            sections["critical_instruction_absolute_adherence"],
            sections["critical_json_output_instructions"],
            sections["json_schema"],
            sections["security_analysis"],
            sections["token_optimization"],
            sections["testing_strategy"],
            sections["ai_reasoning_quality"],
        ]
        
        # Filter out empty sections and join
        combined_prompt_parts = [part for part in prioritized_sections if part]
        reconstructed_prompt = "\n\n".join(combined_prompt_parts)

        # Final truncation if still too long
        final_optimized_prompt = self.tokenizer.truncate_to_token_limit(
            reconstructed_prompt, max_tokens, 
            truncation_indicator="\n\n[...further details truncated for token limits...]"
        )
        
        return final_optimized_prompt