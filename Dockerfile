# Stage 1: Download SentenceTransformer model
FROM python:3.11-slim AS model_downloader
WORKDIR /tmp

# Set Hugging Face cache directories
ENV HF_HOME=/tmp/hf_cache
ENV TRANSFORMERS_CACHE=/tmp/hf_cache/transformers
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/hf_cache/sentence_transformers

# Create cache directories
RUN mkdir -p ${HF_HOME}/transformers && \
    mkdir -p ${HF_HOME}/sentence_transformers

# Install sentence-transformers
RUN pip install --no-cache-dir sentence-transformers

# Download and save the model to our specified cache location
# Using a single-line command to avoid Docker parsing issues
RUN python -c "from sentence_transformers import SentenceTransformer; import os; os.makedirs('/tmp/hf_cache/transformers/sentence-transformers_all-MiniLM-L6-v2', exist_ok=True); model = SentenceTransformer('all-MiniLM-L6-v2'); model.save('/tmp/hf_cache/transformers/sentence-transformers_all-MiniLM-L6-v2')"


# Stage 2: Build application dependencies
FROM python:3.11-slim AS builder
WORKDIR /app

# Install production dependencies
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy the cached model from the model_downloader stage
COPY --from=model_downloader /tmp/hf_cache/transformers /home/appuser/.cache/huggingface/transformers


# Stage 3: Final image
FROM python:3.11-slim AS final

# Create non-root user
RUN useradd -m -u 1000 appuser
WORKDIR /home/appuser

# Copy installed packages and model cache
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /home/appuser/.cache /home/appuser/.cache

# Ensure proper ownership
RUN chown -R appuser:appuser /home/appuser/.cache

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.maxUploadSize=1028"]