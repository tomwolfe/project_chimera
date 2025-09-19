# src/persona/routing.py
"""
Dynamic persona routing system that selects appropriate personas
based on prompt analysis and intermediate results.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Set, Optional, Any, Tuple, TYPE_CHECKING
import re
import logging
from functools import lru_cache

from src.models import PersonaConfig
from src.constants import SELF_ANALYSIS_PERSONA_SEQUENCE
from src.utils.prompt_analyzer import PromptAnalyzer

logger = logging.getLogger(__name__)

# NEW: Use TYPE_CHECKING to avoid circular import at runtime
if TYPE_CHECKING:
    from src.persona_manager import PersonaManager


def calculate_persona_performance(turn):
    """Calculate a performance metric for a persona based on their contribution quality"""
    # Simplified implementation - would use more sophisticated metrics in production
    score = 0

    # Check if the persona addressed the current topic
    if "topic" in turn.context and turn.context["topic"] in turn.content:
        score += 0.3

    # Check if the response was concise and relevant
    if len(turn.content.split()) < 100:
        score += 0.2

    return score


def select_personas_by_weight(weights):
    """Select personas based on their current weights"""
    total_weight = sum(weights.values())
    # The original code for select_personas_by_weight was not provided in the diff,
    # but the diff implies it exists. Assuming it was meant to be added or modified.
    # For now, I'll provide a basic implementation that selects based on weights.
    # This function is referenced in core.py's diff, so it needs to exist.
    if total_weight == 0:
        return []  # No personas to select if total weight is zero

    # Example: Select top N personas or sample based on weights
    # For a simple selection, let's just return all personas sorted by weight
    # or a fixed number of top personas.

    # A simple weighted random selection could be:
    # persona_names = list(weights.keys())
    # persona_probs = [w / total_weight for w in weights.values()]
    # selected_persona_names = np.random.choice(persona_names, size=min(3, len(persona_names)), p=persona_probs, replace=False)
    # return [persona_name for persona_name in selected_persona_names]

    # For now, a simpler approach: return all personas sorted by weight (descending)
    sorted_personas = sorted(weights.items(), key=lambda item: item[1], reverse=True)
    return [name for name, _ in sorted_personas]


class PersonaRouter:
    """Determines the optimal sequence of personas for a given prompt."""

    def __init__(
        self,
        all_personas: Dict[str, PersonaConfig],
        persona_sets: Dict[str, List[str]],
        prompt_analyzer: PromptAnalyzer,
        # NEW: Add persona_manager parameter with Optional type hint
        persona_manager: Optional["PersonaManager"] = None,
    ):
        self.all_personas = all_personas
        self.persona_sets = persona_sets
        self.prompt_analyzer = prompt_analyzer
        # NEW: Store persona_manager
        self.persona_manager = persona_manager

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.persona_embeddings = self._generate_persona_embeddings()

        self.trigger_keywords = {
            "Security_Auditor": [
                "vulnerab",
                "security",
                "exploit",
                "hack",
                "auth",
                "encrypt",
                "threat",
                "risk",
                "malware",
                "penetration",
                "compliance",
                "firewall",
                "ssl",
                "tls",
                "attack vector",
                "data breach",
            ],
            "Test_Engineer": [
                "test",
                "cover",
                "unit",
                "integration",
                "bug",
                "error",
                "quality",
                "qa",
                "defect",
                "debug",
                "validate",
                "assertion",
                "failure",
                "edge case",
                "pytest",
                "unittest",
            ],
            "DevOps_Engineer": [
                "deploy",
                "ci/cd",
                "pipeline",
                "infra",
                "monitor",
                "cloud",
                "docker",
                "k8s",
                "ops",
                "server",
                "automation",
                "release",
                "scalability",
                "reliability",
                "performance",
                "logging",
                "alerting",
            ],
            "Code_Architect": [
                "architect",
                "design",
                "pattern",
                "scalab",
                "perform",
                "modular",
                "refactor",
                "system",
                "structure",
                "database",
                "api",
                "framework",
                "codebase",
                "maintainability",
                "technical debt",
                "separation of concerns",
                "microservice",
                "monolith",
                "backend",
                "frontend",
            ],
            "Constructive_Critic": [
                "improve",
                "refine",
                "optimize",
                "recommend",
                "suggest",
                "enhanc",
                "fix",
                "best practice",
            ],
            "Skeptical_Generator": [
                "risk",
                "flaw",
                "limitation",
                "vulnerab",
                "bottleneck",
                "edge case",
                "failure point",
                "concern",
                "doubt",
            ],
        }

    def _generate_persona_embeddings(self) -> Dict[str, Any]:
        """Generates embeddings for all persona descriptions for semantic routing."""
        embeddings = {}
        for name, config in self.all_personas.items():
            if config.description:
                embeddings[name] = self.model.encode([config.description])[0]
        return embeddings

    def _should_include_test_engineer(
        self,
        prompt_lower: str,
        context_analysis_results: Optional[Dict[str, Any]],
        domain: str,
    ) -> bool:
        """
        Determine if Test_Engineer persona is needed based on prompt, context, and domain.
        For 'Self-Improvement' domain, Test_Engineer is always relevant.
        """
        if domain.lower() == "self-improvement":
            return True  # Test_Engineer is always relevant for self-improvement

        testing_keywords = [
            "test",
            "unit test",
            "integration test",
            "e2e test",
            "test coverage",
            "bug",
            "fix bug",
            "debug",
            "qa",
            "quality assurance",
            "validate",
            "verify",
            "assertion",
            "test case",
            "test suite",
            "pytest",
            "unittest",
            "robustness",
            "maintainability",
        ]
        if any(keyword in prompt_lower for keyword in testing_keywords):
            return True

        if context_analysis_results and context_analysis_results.get("relevant_files"):
            for file_path, _ in context_analysis_results["relevant_files"]:
                if (
                    "test" in file_path.lower()
                    or "spec" in file_path.lower()
                    or file_path.startswith("tests/")
                ):
                    return True

        return False

    def _apply_dynamic_adjustment(
        self,
        sequence: List[str],
        intermediate_results: Optional[Dict[str, Any]],
        prompt_lower: str,
        domain: str,
        context_analysis_results: Optional[Dict[str, Any]],
    ) -> List[str]:
        """Apply dynamic adjustments to persona sequence based on intermediate results quality metrics."""
        if not intermediate_results:
            intermediate_results = {}

        context_analysis_output = intermediate_results.get(
            "Context_Aware_Assistant_Output"
        )
        if context_analysis_output and isinstance(context_analysis_output, dict):
            key_modules = context_analysis_output.get("key_modules", [])
            security_concerns = context_analysis_output.get("security_concerns", [])

            avg_code_quality = 1.0
            avg_complexity = 0.0
            if key_modules:
                avg_code_quality = sum(
                    m.get("code_quality_score", 1.0) for m in key_modules
                ) / len(key_modules)
                avg_complexity = sum(
                    m.get("complexity_score", 0.0) for m in key_modules
                ) / len(key_modules)

            if security_concerns:
                self._insert_persona_before_arbitrator(sequence, "Security_Auditor")
                logger.info(
                    "Prioritized Security_Auditor due to security concerns from context analysis."
                )

            if avg_code_quality < 0.7 or avg_complexity > 0.7:
                self._insert_persona_before_arbitrator(sequence, "Code_Architect")
                logger.info(
                    "Prioritized Code_Architect due to low code quality/maintainability or high complexity from context analysis."
                )

        # Ensure Code_Architect is only present if explicitly needed for architectural concerns
        # and not just as a default for Software Engineering if the prompt is not architectural.
        # This logic is now more robust to avoid misclassification.
        if "Code_Architect" in sequence:
            # Check if the prompt has strong architectural keywords
            architectural_keywords_in_prompt = any(
                term in prompt_lower
                for term in [
                    "architecture",
                    "design",
                    "structure",
                    "refactor",
                    "codebase",
                    "system design",
                    "modularity",
                    "scalability",
                    "maintainability",
                ]
            )
            # Check if context analysis explicitly highlighted architectural patterns or key modules
            context_has_architectural_focus = context_analysis_results and (
                context_analysis_results.get("architectural_patterns")
                or context_analysis_results.get("key_modules")
            )

            # If Code_Architect is in the sequence but no strong architectural focus is detected, remove it.
            if not (
                architectural_keywords_in_prompt or context_has_architectural_focus
            ):
                if "Code_Architect" in sequence:
                    sequence.remove("Code_Architect")
                    logger.info(
                        "Removed Code_Architect from sequence as no strong architectural context/keywords detected."
                    )

            # Handle potential misclassification for "building architecture" vs "software architecture"
            # This is a more nuanced check to prevent removing Code_Architect for valid software architecture prompts.
            building_arch_terms = [
                "building",
                "construction",
                "physical",
                "blueprint",
                "house",
                "floor plan",
            ]
            software_arch_terms = [
                "software",
                "code",
                "system",
                "api",
                "database",
                "backend",
                "frontend",
            ]
            building_count = sum(
                1 for term in building_arch_terms if term in prompt_lower
            )
            software_count = sum(
                1 for term in software_arch_terms if term in prompt_lower
            )
            if (
                building_count > software_count
                and building_count >= 2
                and software_count == 0
            ):
                logger.warning(
                    f"Misclassification detected: Building architecture prompt likely triggered Code_Architect. Removing it."
                )
                if "Code_Architect" in sequence:
                    sequence.remove("Code_Architect")
                if (
                    "Creative_Thinker" not in sequence
                    and "Creative_Thinker" in self.all_personas
                ):
                    self._insert_persona_before_arbitrator(sequence, "Creative_Thinker")

        adjusted_sequence = sequence.copy()

        if domain == "Software Engineering" or domain == "Self-Improvement":
            if (
                "Test_Engineer" in adjusted_sequence
                and not self._should_include_test_engineer(
                    prompt_lower, context_analysis_results, domain
                )
            ):
                adjusted_sequence.remove("Test_Engineer")
                logger.info(
                    "Removed Test_Engineer from sequence as no testing context/keywords detected."
                )
            elif (
                "Test_Engineer" not in adjusted_sequence
                and self._should_include_test_engineer(
                    prompt_lower, context_analysis_results, domain
                )
            ):
                self._insert_persona_before_arbitrator(
                    adjusted_sequence, "Test_Engineer"
                )
                logger.info(
                    "Added Test_Engineer to sequence due to testing context/keywords detected."
                )

        reasoning_quality_metrics = intermediate_results.get(
            "Self_Improvement_Metrics", {}
        ).get("reasoning_quality", {})
        if reasoning_quality_metrics:
            schema_failures = reasoning_quality_metrics.get(
                "schema_validation_failures_count", 0
            )
            content_misalignments = reasoning_quality_metrics.get(
                "content_misalignment_warnings", 0
            )
            unresolved_conflict = reasoning_quality_metrics.get(
                "unresolved_conflict_present", False
            )

            if (
                schema_failures > 0 or content_misalignments > 0
            ) and "Constructive_Critic" not in adjusted_sequence:
                self._insert_persona_before_arbitrator(
                    adjusted_sequence, "Constructive_Critic"
                )
                logger.info(
                    "Prioritized Constructive_Critic due to schema failures or content misalignment."
                )

            if unresolved_conflict and "Devils_Advocate" not in adjusted_sequence:
                self._insert_persona_before_arbitrator(
                    adjusted_sequence, "Devils_Advocate"
                )
                logger.info("Prioritized Devils_Advocate due to unresolved conflicts.")

        return adjusted_sequence

    def _insert_persona_before_arbitrator(self, sequence: List[str], persona: str):
        """Insert persona before the Impartial_Arbitrator in the sequence if not already present."""
        if persona in sequence:
            return

        arbitrator_index = len(sequence)
        if "Impartial_Arbitrator" in sequence:
            arbitrator_index = sequence.index("Impartial_Arbitrator")
        elif "Self_Improvement_Analyst" in sequence:  # Standardized to snake_case
            arbitrator_index = sequence.index(
                "Self_Improvement_Analyst"
            )  # Standardized to snake_case
        elif "General_Synthesizer" in sequence:
            arbitrator_index = sequence.index("General_Synthesizer")

        sequence.insert(arbitrator_index, persona)
        logger.debug(
            f"Inserted persona '{persona}' before Arbitrator/Synthesizer at index {arbitrator_index}."
        )

    def determine_persona_sequence(
        self,
        prompt: str,
        domain: str,
        intermediate_results: Optional[Dict[str, Any]] = None,
        context_analysis_results: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Determine the optimal sequence of personas for processing the prompt.
        """
        prompt_lower = prompt.lower()

        if self.prompt_analyzer.is_self_analysis_prompt(prompt):
            logger.info(
                "Detected self-analysis prompt. Applying dynamic persona sequence from 'Self-Improvement' set."
            )

            base_sequence = self.persona_sets.get(
                "Self-Improvement", SELF_ANALYSIS_PERSONA_SEQUENCE
            ).copy()

            if (
                any(
                    kw in prompt_lower
                    for kw in [
                        "security",
                        "vulnerability",
                        "exploit",
                        "authentication",
                        "threat",
                        "risk",
                    ]
                )
                and "Security_Auditor" not in base_sequence
            ):
                self._insert_persona_before_arbitrator(
                    base_sequence, "Security_Auditor"
                )
                logger.info(
                    "Self-analysis prompt is security-focused. Added Security_Auditor."
                )

            if (
                any(
                    kw in prompt_lower
                    for kw in [
                        "performance",
                        "efficiency",
                        "scalability",
                        "devops",
                        "ci/cd",
                        "deployment",
                    ]
                )
                and "DevOps_Engineer" not in base_sequence
            ):
                self._insert_persona_before_arbitrator(base_sequence, "DevOps_Engineer")
                logger.info(
                    "Self-analysis prompt is performance/DevOps-focused. Added DevOps_Engineer."
                )

            if (
                any(
                    kw in prompt_lower
                    for kw in [
                        "maintainability",
                        "readability",
                        "structure",
                        "refactor",
                        "clean code",
                    ]
                )
                and "Code_Architect" not in base_sequence
            ):
                if "Code_Architect" in base_sequence:
                    base_sequence.remove("Code_Architect")
                    base_sequence.insert(0, "Code_Architect")
                else:
                    self._insert_persona_before_arbitrator(
                        base_sequence, "Code_Architect"
                    )
                logger.info(
                    "Self-analysis prompt is maintainability/structure-focused. Prioritized Code_Architect."
                )

            # Standardize persona name to snake_case
            if "SelfImprovementAnalyst" in base_sequence:
                base_sequence.remove("SelfImprovementAnalyst")
            if "Self_Improvement_Analyst" not in base_sequence:
                base_sequence.append("Self_Improvement_Analyst")

            if "Impartial_Arbitrator" in base_sequence:
                base_sequence.remove("Impartial_Arbitrator")
                # Insert before Self_Improvement_Analyst if it's present
                try:
                    analyst_idx = base_sequence.index("Self_Improvement_Analyst")
                    base_sequence.insert(analyst_idx, "Impartial_Arbitrator")
                except ValueError:
                    # If Self_Improvement_Analyst is not in base_sequence (shouldn't happen with logic above),
                    # append Arbitrator at the end.
                    base_sequence.append("Impartial_Arbitrator")

            if "Devils_Advocate" in base_sequence:
                base_sequence.remove("Devils_Advocate")

            insert_pos_for_advocate = len(base_sequence)
            if "Impartial_Arbitrator" in base_sequence:
                insert_pos_for_advocate = base_sequence.index("Impartial_Arbitrator")
            elif (
                "Self_Improvement_Analyst" in base_sequence
            ):  # Standardized to snake_case
                insert_pos_for_advocate = base_sequence.index(
                    "Self_Improvement_Analyst"  # Standardized to snake_case
                )

            if (
                "Constructive_Critic" in base_sequence
                and base_sequence.index("Constructive_Critic") < insert_pos_for_advocate
            ):
                critic_idx = base_sequence.index("Constructive_Critic")
                base_sequence.insert(critic_idx + 1, "Devils_Advocate")
            else:
                base_sequence.insert(insert_pos_for_advocate, "Devils_Advocate")

            final_sequence = base_sequence
            logger.info(f"Self-analysis persona sequence: {final_sequence}")

        else:
            if self.persona_embeddings:
                prompt_embedding = self.model.encode([prompt])[0]
                semantic_scores = {}
                for p_name, p_embedding in self.persona_embeddings.items():
                    semantic_scores[p_name] = np.dot(prompt_embedding, p_embedding) / (
                        np.linalg.norm(prompt_embedding) * np.linalg.norm(p_embedding)
                    )

                domain_personas = self.persona_sets.get(domain, [])
                for p_name in domain_personas:
                    semantic_scores[p_name] = semantic_scores.get(p_name, 0.0) + 0.2

                top_semantic_personas = sorted(
                    semantic_scores.items(), key=lambda x: x[1], reverse=True
                )[:5]
                initial_semantic_sequence = [
                    p[0] for p in top_semantic_personas if p[0] in self.all_personas
                ]

                base_sequence = []
                for p_name in initial_semantic_sequence:
                    if p_name not in base_sequence:
                        base_sequence.append(p_name)
                for p_name in self.persona_sets.get(domain, []):
                    if p_name not in base_sequence:
                        base_sequence.append(p_name)

                synthesis_persona = (
                    "Impartial_Arbitrator"
                    if domain == "Software Engineering"
                    else "General_Synthesizer"
                )
                if synthesis_persona not in base_sequence:
                    base_sequence.append(synthesis_persona)
                else:
                    base_sequence.remove(synthesis_persona)
                    base_sequence.append(synthesis_persona)

                logger.info(
                    f"Initial semantic-driven persona sequence: {base_sequence}"
                )
            else:
                if domain not in self.persona_sets:
                    logger.warning(
                        f"Domain '{domain}' not found in persona_sets. Falling back to 'General' sequence."
                    )
                    domain = "General"
                base_sequence = self.persona_sets.get(domain, [])
                if not base_sequence:
                    logger.error(
                        "No valid persona sequence found. Using minimal fallback."
                    )
                    base_sequence = [
                        "Visionary_Generator",
                        "Skeptical_Generator",
                        "Impartial_Arbitrator",
                    ]

            final_sequence = base_sequence.copy()

        final_sequence = self._apply_dynamic_adjustment(
            final_sequence,
            intermediate_results,
            prompt_lower,
            domain,
            context_analysis_results,
        )

        if context_analysis_results:
            relevant_files = context_analysis_results.get("relevant_files", [])
            test_file_count = sum(
                1 for file_path, _ in relevant_files if file_path.startswith("tests/")
            )
            code_file_count = sum(
                1
                for file_path, _ in relevant_files
                if file_path.endswith((".py", ".js", ".ts", ".java", ".go"))
            )

            if test_file_count > 3 and "Test_Engineer" not in final_sequence:
                self._insert_persona_before_arbitrator(final_sequence, "Test_Engineer")

            if code_file_count > 5:
                if "Code_Architect" not in final_sequence:
                    self._insert_persona_before_arbitrator(
                        final_sequence, "Code_Architect"
                    )
                if "Security_Auditor" not in final_sequence:
                    self._insert_persona_before_arbitrator(
                        final_sequence, "Security_Auditor"
                    )

        seen = set()
        unique_sequence = []
        for persona in final_sequence:
            if persona not in seen:
                unique_sequence.append(persona)
                seen.add(persona)

        return unique_sequence

    def _analyze_prompt_complexity(self, prompt: str) -> Dict[str, Any]:
        """Analyze prompt complexity with domain-specific weighting."""
        return self.prompt_analyzer.analyze_complexity(prompt)
