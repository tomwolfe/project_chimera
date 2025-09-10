# Stage 1: Download SentenceTransformer model
FROM python:3.11-slim AS model_downloader
WORKDIR /tmp

# Set Hugging Face cache directories for model download
ENV HF_HOME=/tmp/hf_cache
ENV TRANSFORMERS_CACHE=/tmp/hf_cache/transformers
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/hf_cache/sentence_transformers
# NEW: Explicitly disable tokenizers parallelism to avoid deadlocks on fork
ENV TOKENIZERS_PARALLELISM=false

# Create cache directories
RUN mkdir -p ${HF_HOME}/transformers && \
    mkdir -p ${HF_HOME}/sentence_transformers

# Install sentence-transformers
RUN pip install --no-cache-dir sentence-transformers

# Download and save the model to our specified cache location
# Using a single-line command to avoid Docker parsing issues
RUN python -c "from sentence_transformers import SentenceTransformer; import os; os.makedirs('/tmp/hf_cache/transformers/sentence-transformers_all-MiniLM-L6-v2', exist_ok=True); model = SentenceTransformer('all-MiniLM-L6-v2'); model.save('/tmp/hf_cache/transformers/sentence-transformers_all-MiniLM-L6-v2')"


# Stage 2: Final image
FROM python:3.11-slim AS final

# Create non-root user
RUN useradd -m -u 1000 appuser # Create appuser
WORKDIR /home/appuser

# Copy production requirements
COPY --chown=appuser:appuser requirements-prod.txt .

# Install production dependencies as appuser
USER appuser
RUN pip install --no-cache-dir -r requirements-prod.txt

# Switch back to root temporarily if needed for system-wide installs, then back to appuser
# For now, assume all installs can be done as appuser or are already handled.

# Copy the cached model from the model_downloader stage
# This ensures the SentenceTransformer model is available offline within the appuser's cache
COPY --from=model_downloader /tmp/hf_cache/transformers /home/appuser/.cache/huggingface/transformers

# Ensure proper ownership of the cache directory for the non-root user
RUN chown -R appuser:appuser /home/appuser/.cache

# Copy application code
COPY --chown=appuser:appuser . .

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.maxUploadSize=128028"]