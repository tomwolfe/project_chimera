# Stage 1: Download SentenceTransformer model
FROM python:3.11-slim AS model_downloader
WORKDIR /tmp

# Set Hugging Face cache directories for model download
ENV HF_HOME=/tmp/hf_cache
ENV TRANSFORMERS_CACHE=/tmp/hf_cache/transformers
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/hf_cache/sentence_transformers
# NEW: Explicitly disable tokenizers parallelism to avoid deadlocks on fork
ENV TOKENIZERS_PARALLELISM=false

# Create cache directories (as root in this stage)
RUN mkdir -p ${HF_HOME}/transformers && \
    mkdir -p ${HF_HOME}/sentence_transformers

# Install sentence-transformers (as root in this stage)
RUN pip install --no-cache-dir sentence-transformers

# Download and save the model to our specified cache location (as root in this stage)
RUN python -c "from sentence_transformers import SentenceTransformer; import os; os.makedirs('/tmp/hf_cache/transformers/sentence-transformers_all-MiniLM-L6-v2', exist_ok=True); model = SentenceTransformer('all-MiniLM-L6-v2'); model.save('/tmp/hf_cache/transformers/sentence-transformers_all-MiniLM-L6-v2')"


# Stage 2: Final image
FROM python:3.11-slim AS final

# Create non-root user (as root)
RUN useradd -m -u 1000 appuser

# Set WORKDIR for root operations (will be changed to appuser later)
WORKDIR /home/appuser

# Copy production requirements (as root)
COPY requirements-prod.txt .

# Install production dependencies (as root, so they are in system PATH and accessible to appuser)
RUN pip install --no-cache-dir -r requirements-prod.txt

# Create the .cache/huggingface directory structure for appuser and ensure appuser owns it
# This must be done as root.
RUN mkdir -p /home/appuser/.cache/huggingface/transformers && \
    chown -R appuser:appuser /home/appuser/.cache

# Copy the cached model from the model_downloader stage
# Use --chown to ensure appuser owns the copied files immediately.
# This command is run as root, so it has permissions to change ownership.
COPY --from=model_downloader --chown=appuser:appuser /tmp/hf_cache/transformers /home/appuser/.cache/huggingface/transformers

# Copy application code (as root, but chown to appuser)
COPY --chown=appuser:appuser . .

# Switch to non-root user for running the application
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.maxUploadSize=128028"]