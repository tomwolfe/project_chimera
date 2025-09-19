# src/rag_system.py
import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class KnowledgeRetriever:
    def __init__(
        self,
        raw_file_contents: Dict[str, str],
        cache_dir: str = "./.cache/rag_embeddings",
    ):
        self.raw_file_contents = raw_file_contents
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.file_embeddings = self._load_or_compute_embeddings()

    def _load_or_compute_embeddings(self) -> Dict[str, Any]:
        # For simplicity, recompute every time for now, but a real system would use hashing
        # to check if raw_file_contents changed before recomputing.
        logger.info("Computing RAG embeddings for codebase files...")
        file_paths = list(self.raw_file_contents.keys())
        file_contents = list(self.raw_file_contents.values())
        if not file_contents:
            return {}
        embeddings_list = self.model.encode(file_contents)
        return dict(zip(file_paths, embeddings_list))

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        if not self.file_embeddings:
            return []
        query_embedding = self.model.encode([query])[0]

        similarities = []
        for file_path, embedding in self.file_embeddings.items():
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append((file_path, similarity))

        sorted_docs = sorted(similarities, key=lambda x: x[1], reverse=True)

        retrieved_snippets = []
        for file_path, score in sorted_docs[:top_k]:
            if score > 0.5:  # Only include sufficiently relevant documents
                content = self.raw_file_contents.get(file_path, "")
                # Simple snippet extraction (e.g., first 200 chars)
                snippet = content[:200] + "..." if len(content) > 200 else content
                retrieved_snippets.append(f"File: {file_path}\nContent: {snippet}")
        return retrieved_snippets


class RagOrchestrator:
    def __init__(self, retriever: KnowledgeRetriever, persona_manager: Any):
        self.retriever = retriever
        self.persona_manager = (
            persona_manager  # Assuming persona_manager has access to prompt templates
        )

    def generate_rag_context(self, user_query: str, persona_name: str) -> str:
        # Retrieve relevant documents based on the query and persona's focus
        retrieved_docs = self.retriever.retrieve(user_query)
        if not retrieved_docs:
            return ""

        context_str = "\n".join(retrieved_docs)
        return f"Retrieved relevant codebase context:\n{context_str}\n\n"


# Example Usage (requires retriever and prompt_manager instances)
# from src.persona_manager import PersonaManager
# from src.config.settings import ChimeraSettings
# from src.utils.prompt_analyzer import PromptAnalyzer
#
# if __name__ == "__main__":
#     # Dummy setup for demonstration
#     mock_raw_files = {
#         "src/core.py": "class SocraticDebate:\n    def run_debate(self):\n        # core logic here\n        pass",
#         "src/llm_provider.py": "class GeminiProvider:\n    def generate(self, prompt):\n        # LLM call logic\n        pass",
#         "docs/project_chimera_context.md": "Project Chimera is a self-improving AI."
#     }
#     mock_settings = ChimeraSettings()
#     mock_prompt_analyzer = PromptAnalyzer(mock_settings.domain_keywords)
#     mock_persona_manager = PersonaManager(mock_settings.domain_keywords, prompt_analyzer=mock_prompt_analyzer)
#
#     retriever = KnowledgeRetriever(raw_file_contents=mock_raw_files)
#     rag_orchestrator = RagOrchestrator(retriever, mock_persona_manager)
#
#     query = "How does SocraticDebate work?"
#     context = rag_orchestrator.generate_rag_context(query, "Self_Improvement_Analyst")
#     print(context)
