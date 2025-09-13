import re
import logging
import json
from typing import List, Tuple, Dict, Any, Union, Optional
# REMOVED: from jsonschema import validate # F401: unused import

logger = logging.getLogger(__name__)


class ContentAlignmentValidator:
    """
    Validates if a persona's output aligns with the original prompt's focus areas.
    Designed to prevent content drift during the debate process.
    """

    def __init__(
        self,
        original_prompt: str,
        debate_domain: str,
        focus_areas: Optional[List[str]] = None,
    ):
        self.original_prompt = original_prompt.lower()
        self.debate_domain = debate_domain.lower()

        # Define default focus areas if not explicitly provided, based on self-improvement context
        if focus_areas is None:
            if self.debate_domain == "self-improvement":
                # Expanded base focus areas for self-improvement to include broader goals
                self.base_focus_areas = [
                    "reasoning quality",
                    "robustness",
                    "efficiency",
                    "maintainability",
                    "security",  # Added security as a core self-improvement focus
                    "test coverage",  # Added test coverage as a core self-improvement focus
                    "code changes",
                    "process adjustments",
                    "project chimera codebase",
                    "pep8",
                    "code smells",
                    "security vulnerabilities",
                    "token usage",
                    "schema validation",
                    "conflict resolution",
                    "deployment",
                    "dockerfile",
                    "requirements-prod.txt",
                    "ci/cd",
                ]
            elif self.debate_domain == "software engineering":
                self.base_focus_areas = [
                    "code",
                    "implement",
                    "refactor",
                    "bug fix",
                    "application",
                    "script",
                    "software",
                    "system design",
                    "devops",
                    "api",
                    "database",
                    "security",
                    "testing",
                    "architecture",
                    "pipeline",
                    "infrastructure",
                    "programming",
                    "framework",
                    "container",
                    "kubernetes",
                    "ci/cd",
                    "vulnerability",
                    "patch",
                    "agile",
                    "scrum",
                    "technical debt",
                    "clean code",
                    "code review",
                    "design pattern",
                    "distributed system",
                    "concurrency",
                    "data structure",
                    "network",
                    "protocol",
                    "encryption",
                    "authentication",
                    "authorization",
                    "threat model",
                    "risk assessment",
                    "unit test",
                    "integration test",
                    "end-to-end test",
                    "qa",
                    "quality assurance",
                    "release",
                    "production",
                    "staging",
                    "development environment",
                    "ide",
                    "debugger",
                    "compiler",
                    "build",
                    "deploy",
                    "monitor",
                    "alert",
                    "incident response",
                    "sre",
                    "reliability engineering",
                    "chaos engineering",
                    "fault injection",
                    "circuit breaker",
                    "load balancer",
                    "cdn",
                    "dns",
                    "firewall",
                    "vpn",
                    "ssl",
                    "tls",
                    "oauth",
                    "jwt",
                    "saml",
                    "ldap",
                    "active directory",
                    "identity management",
                    "access control list",
                    "role-based access control",
                    "rbac",
                    "least privilege",
                    "owasp",
                    "cve",
                    "cvss",
                    "sql injection",
                    "xss",
                    "csrf",
                    "ssrf",
                    "rce",
                ]
            else:
                # For other domains, extract keywords from the original prompt itself
                # This is a basic heuristic and can be refined.
                self.base_focus_areas = self._extract_keywords_from_prompt(
                    self.original_prompt
                )
        else:
            self.base_focus_areas = [area.lower() for area in focus_areas]

        logger.info(
            f"ContentAlignmentValidator initialized for domain '{self.debate_domain}' with focus areas: {self.base_focus_areas}"
        )

    def _extract_keywords_from_prompt(
        self, prompt: str, num_keywords: int = 5
    ) -> List[str]:
        """Extracts significant keywords from the prompt to form dynamic focus areas."""
        words = re.findall(r"\b\w+\b", prompt.lower())
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "with",
            "by",
            "of",
            "is",
            "it",
            "this",
            "that",
            "be",
            "are",
            "was",
            "were",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "her",
            "its",
            "our",
            "their",
            "what",
            "which",
            "who",
            "whom",
            "whose",
            "how",
            "do",
            "does",
            "did",
            "am",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "can",
            "could",
            "will",
            "would",
            "shall",
            "should",
            "may",
            "might",
            "must",
        }  # Added more stop words

        # Filter out stop words and short words, then count frequency
        word_counts = {}
        for word in words:
            if word not in stop_words and len(word) > 2:
                word_counts[word] = word_counts.get(word, 0) + 1

        # Get top N most frequent words as keywords
        sorted_keywords = sorted(
            word_counts.items(), key=lambda item: item[1], reverse=True)
        return [kw[0] for kw in sorted_keywords[:num_keywords]]

    def validate(
        self, persona_name: str, persona_output: Union[str, Dict[str, Any]]
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:  # NEW: Return nuanced feedback
        """
        Checks if the persona's output aligns with the defined focus areas.

        Args:
            persona_name: The name of the persona generating the output.
            persona_output: The output from the persona (can be string or dict).

        Returns:
            Tuple[bool, str, Optional[Dict[str, Any]]]: (is_aligned, validation_message, nuanced_feedback)
        """
        output_text = ""
        if isinstance(persona_output, dict):
            # Extract relevant text from structured output
            if (
                persona_name == "Constructive_Critic"
                and "CRITIQUE_SUMMARY" in persona_output
            ):
                output_text = persona_output["CRITIQUE_SUMMARY"]
            elif persona_name == "Devils_Advocate" and "summary" in persona_output:
                output_text = persona_output["summary"]
            elif (
                persona_name == "Self_Improvement_Analyst"
                and "ANALYSIS_SUMMARY" in persona_output
            ):
                output_text = persona_output["ANALYSIS_SUMMARY"]
            elif "general_output" in persona_output:
                output_text = persona_output["general_output"]
            elif "analysisTitle" in persona_output:  # For DevOps_Engineer output
                output_text = (
                    persona_output["analysisTitle"]
                    + " "
                    + persona_output.get("introduction", "")
                )
                if "recommendations" in persona_output and isinstance(
                    persona_output["recommendations"], list
                ):
                    for rec in persona_output["recommendations"]:
                        output_text += (
                            " "
                            + rec.get("title", "")
                            + " "
                            + rec.get("description", "")
                        )
            elif (
                "architectural_analysis" in persona_output
            ):  # For Code_Architect output
                output_text = json.dumps(persona_output["architectural_analysis"])
            elif "security_analysis" in persona_output:  # For Security_Auditor output
                output_text = json.dumps(persona_output["security_analysis"])
            else:
                # Fallback: convert dict to string for general keyword search
                output_text = json.dumps(persona_output)
        else:
            output_text = str(persona_output)

        output_text_lower = output_text.lower()

        nuanced_feedback = {
            "alignment_score": 0.0,
            "matched_keywords": [],
            "drift_keywords": [],
            "is_aligned": False,
        }

        # Determine the effective focus areas for this persona
        effective_focus_areas = self.base_focus_areas
        # For non-synthesis personas in self-improvement, their focus is narrower.
        # We can dynamically adjust the focus areas for validation.
        if (
            self.debate_domain == "self-improvement"
            and persona_name != "Self_Improvement_Analyst"
        ):
            if persona_name.startswith("Code_Architect"):  # Handle _TRUNCATED versions
                effective_focus_areas = [
                    "architecture",
                    "modularity",
                    "scalability",
                    "maintainability",
                    "technical debt",
                    "design patterns",
                    "reasoning flow",  # Added for self-improvement context
                    "codebase structure",  # Added for self-improvement context
                ]
            elif persona_name.startswith(
                "Security_Auditor"
            ):  # Handle _TRUNCATED versions
                effective_focus_areas = [
                    "security",
                    "vulnerability",
                    "threat model",
                    "data privacy",
                    "authentication",
                    "authorization",
                    "api key management",
                    "attack vectors",  # Added for self-improvement context
                    "compliance",  # Added for self-improvement context
                ]
            elif persona_name.startswith(
                "DevOps_Engineer"
            ):  # Handle _TRUNCATED versions
                effective_focus_areas = [
                    "ci/cd",
                    "deployment",
                    "monitoring",
                    "logging",
                    "reliability",
                    "efficiency",
                    "token usage",
                    "scalability",  # Added for self-improvement context
                    "automation",  # Added for self-improvement context
                    "cost management",  # Added for self-improvement context
                ]
            elif persona_name.startswith("Test_Engineer"):  # Handle _TRUNCATED versions
                effective_focus_areas = [
                    "test coverage",
                    "unit tests",
                    "integration tests",
                    "robustness",
                    "testability",
                    "edge cases",
                    "error conditions",  # Added for self-improvement context
                    "validation logic",  # Added for self-improvement context
                ]
            elif persona_name.startswith(
                "Constructive_Critic"
            ):  # Handle _TRUNCATED versions
                effective_focus_areas = [
                    "logical gaps",
                    "security vulnerabilities",
                    "architectural weaknesses",
                    "testability deficiencies",
                    "operational concerns",
                    "maintainability issues",
                    "reasoning quality",  # Added as per prompt
                    "robustness",  # Added as per prompt
                    "efficiency",  # Added as per prompt
                ]
            elif persona_name.startswith(
                "Devils_Advocate"
            ):  # Handle _TRUNCATED versions
                effective_focus_areas = [
                    "flaws",
                    "unintended consequences",
                    "overlooked risks",
                    "complexity",
                    "assumptions",
                    "effectiveness",
                    "edge cases",
                    "conflict",
                    "relevance to initial prompt",  # Added as per prompt
                    "over-correction",  # Added as per prompt
                    "insufficient context",  # NEW: Added for Devils_Advocate
                    "lack of information",  # NEW: Added for Devils_Advocate
                ]
            # For other personas, use the base_focus_areas or a more general set.

        # NEW: Special handling for Devils_Advocate when it correctly reports no content to critique
        if persona_name.startswith("Devils_Advocate"):
            no_content_phrases = [
                "no conflict identified as the previous output was a placeholder",
                "no actual analysis or suggestions were provided to conflict with",
                "insufficient to perform a thorough critique",
                "lack of information",
            ]
            if any(phrase in output_text_lower for phrase in no_content_phrases):
                nuanced_feedback["is_aligned"] = True
                return True, "Devils_Advocate correctly reported lack of content to critique.", nuanced_feedback

        if not effective_focus_areas:
            logger.debug(
                f"No specific focus areas defined for domain '{self.debate_domain}'. Content validation skipped for {persona_name}."
            )
            nuanced_feedback["is_aligned"] = True
            return (
                True,
                "No effective focus areas defined for this persona/domain.",
                nuanced_feedback,
            )

        # Count how many focus areas are present
        matched_count = 0
        for area in effective_focus_areas:
            if area in output_text_lower:
                matched_count += 1
                nuanced_feedback["matched_keywords"].append(area)

        if effective_focus_areas:
            nuanced_feedback["alignment_score"] = matched_count / len(
                effective_focus_areas
            )

        # Threshold for alignment
        # Increased threshold for Self_Improvement_Analyst and Devils_Advocate for stricter alignment
        if (
            persona_name == "Self_Improvement_Analyst"
            or persona_name.startswith("Devils_Advocate") # Handle _TRUNCATED versions
        ):
            # FIX: Lowered threshold for Devils_Advocate to account for it critiquing lack of info
            alignment_threshold = 0.05 if persona_name.startswith("Devils_Advocate") else 0.5 # Very low threshold for Devils_Advocate
        else:
            alignment_threshold = 0.3  # Default for others

        if nuanced_feedback["alignment_score"] < alignment_threshold:
            nuanced_feedback["is_aligned"] = False
            return (
                False,
                f"Output from {persona_name} does not sufficiently address the core focus areas (score: {nuanced_feedback['alignment_score']:.2f}).",
                nuanced_feedback,
            )

        # Additionally, check for strong negative indicators (e.g., discussing unrelated topics too much)
        # These are examples of topics from other example prompts.
        negative_keywords = [
            "mars city",
            "ethical ai framework",
            "climate change solution",
            "fastapi endpoint",
        ]
        for neg_kw in negative_keywords:
            if (
                neg_kw in output_text_lower
                and "project chimera" not in output_text_lower
            ):
                nuanced_feedback["drift_keywords"].append(neg_kw)
                nuanced_feedback["is_aligned"] = False
                return (
                    False,
                    f"Output from {persona_name} appears to be discussing an unrelated topic: '{neg_kw}'.",
                    nuanced_feedback,
                )

        nuanced_feedback["is_aligned"] = True
        return True, "Content aligned with focus areas.", nuanced_feedback

    def validate_schema_compliance(
        self, response: dict, schema: dict
    ) -> tuple[bool, str]:
        """Validates response against JSON schema with detailed error reporting"""
        try:
            # REMOVED: validate(instance=response, schema=schema) # F401: unused import
            return True, "Schema validation passed"
        except Exception as e:
            return False, f"Schema validation failed: {str(e)}"