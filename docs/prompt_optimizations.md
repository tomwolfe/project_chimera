# Prompt Optimization Strategies for Self-Improvement Analyst

This document outlines strategies to optimize the token usage and efficiency of the `Self_Improvement_Analyst` persona.

## 1. Reduce Redundancy in Analysis Steps

*   **Identify Overlapping Checks:** Analyze the current analysis process to identify steps that repeatedly gather or process the same information. Consolidate these steps where possible.
*   **Conditional Deep Dives:** Instead of performing exhaustive analysis on every aspect, implement conditional logic. For example, only perform a deep dive into specific code modules if initial high-level scans indicate potential issues in those areas.

## 2. Implement Prompt Truncation Strategies

*   **Contextual Summarization:** When providing context or previous analysis results, summarize them concisely. Focus on the most relevant findings and avoid including verbose details unless explicitly required.
*   **Prioritize Key Metrics:** When requesting analysis of metrics (e.g., code quality, security), explicitly state which metrics are of highest priority to guide the LLM's focus and reduce the amount of data it needs to process.
*   **Example Truncated Prompt Snippet (Conceptual):
    ```python
    """Analyze the provided code snippets, focusing primarily on security vulnerabilities (SQL injection, command injection, hardcoded secrets) and critical formatting errors. Provide actionable recommendations for the top 3 highest impact issues."""
    ```

## 3. Refine Output Structure

*   **Structured Output:** Encourage the persona to output findings in a structured format (like JSON) with clear headings for each area of analysis. This makes parsing and further processing more efficient.
*   **Limit Verbosity:** Instruct the persona to be concise and avoid unnecessary conversational filler. Focus on delivering actionable insights.

## 4. Caching and Memoization

*   **Cache Analysis Results:** If the same code or metrics are analyzed multiple times, consider implementing a caching mechanism to store and retrieve previous analysis results, avoiding redundant LLM calls.

## 5. Persona-Specific Prompt Engineering

*   **System Prompt Tuning:** Review and refine the system prompt for the `Self_Improvement_Analyst` to explicitly guide it towards efficiency and token-consciousness. 
    *   **Example System Prompt Enhancement:**
        ```
        You are the Self-Improvement Analyst. Your goal is to identify the most impactful improvements in Project Chimera. Focus on actionable insights and prioritize efficiency. Be concise and avoid unnecessary verbosity. Aim to minimize token usage while delivering high-quality analysis.
        ```
