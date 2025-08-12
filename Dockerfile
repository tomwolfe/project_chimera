# Stage 1: Download SentenceTransformer model
# This stage is solely responsible for downloading the model and caching it.
# It will only rebuild if the base image or the pip install command changes.
FROM python:3.11-slim AS model_downloader
WORKDIR /tmp
# Install sentence-transformers here, separately from app dependencies
RUN pip install --no-cache-dir sentence-transformers
# Download the model. It will be cached in /root/.cache/huggingface/transformers
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Stage 2: Build application dependencies and copy the cached model
# This stage installs your application's Python dependencies.
# It will only rebuild if requirements-prod.txt changes.
FROM python:3.11-slim AS builder
WORKDIR /app

# Install production dependencies FIRST to leverage Docker layer caching.
# This layer will only be invalidated if requirements-prod.txt changes.
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy the cached model from the model_downloader stage.
# The model is typically cached in /root/.cache/huggingface/transformers for the root user.
# We copy it to /home/appuser/.cache/huggingface/transformers, which is where the 'appuser'
# (created later) will expect it to be.
COPY --from=model_downloader /root/.cache/huggingface/transformers /home/appuser/.cache/huggingface/transformers

# Stage 3: Final image with non-root user and application code
# This is your final production image.
FROM python:3.11-slim

# Use a non-root user for security
RUN useradd -m -u 1000 appuser
USER appuser

# Copy the installed Python packages from the 'builder' stage.
# This ensures only necessary packages are in the final image, not build tools.
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy the pre-downloaded model cache to the appuser's home directory.
# This ensures the appuser can access the model without re-downloading.
COPY --from=builder /home/appuser/.cache/huggingface/transformers /home/appuser/.cache/huggingface/transformers

# Copy application code. This should be the LAST step that copies source code,
# as it's the most frequently changing part, minimizing cache invalidation.
COPY . .

# Ensure the appuser owns the cached model directory
RUN chown -R appuser:appuser /home/appuser/.cache/huggingface/transformers

# Expose the port Streamlit will run on (Cloud Run default is 8080)
EXPOSE 8080

# Healthcheck for container orchestration
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8080/_stcore/health || exit 1

# Command to run the Streamlit application with production-ready flags
CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.headless", "true", "--server.enableCORS", "true", "--server.enableXsrfProtection", "true", "--server.runOnSave", "false", "--server.fileWatcherType", "none"]