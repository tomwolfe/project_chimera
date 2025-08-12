# Stage 1: Download SentenceTransformer model
# This stage is solely responsible for downloading the model and caching it.
# It will only rebuild if the base image or the pip install command changes.
FROM python:3.11-slim AS model_downloader
WORKDIR /tmp
# Install sentence-transformers here, separately from app dependencies
RUN pip install --no-cache-dir sentence-transformers
# Download the model. It will be cached in /root/.cache/torch/sentence_transformers/all-MiniLM-L6-v2
# The command itself doesn't output much, but it creates the directory.
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

# Corrected COPY path for the SentenceTransformer model.
# The model 'all-MiniLM-L6-v2' is downloaded into a specific subdirectory within ~/.cache/torch/sentence_transformers.
# We need to copy that specific model directory.
# The destination path should match where the 'appuser' will expect it.
COPY --from=model_downloader /root/.cache/torch/sentence_transformers/all-MiniLM-L6-v2 /home/appuser/.cache/torch/sentence_transformers/all-MiniLM-L6-v2

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
# Ensure the destination path matches where sentence-transformers will look for it.
# It will look in ~/.cache/torch/sentence_transformers/all-MiniLM-L6-v2
COPY --from=builder /home/appuser/.cache/torch/sentence_transformers/all-MiniLM-L6-v2 /home/appuser/.cache/torch/sentence_transformers/all-MiniLM-L6-v2

# Copy application code. This should be the LAST step that copies source code,
# as it's the most frequently changing part, minimizing cache invalidation.
COPY . .

# Ensure the appuser owns the cached model directory.
# The parent directory of the model cache needs to be owned by appuser for read/write access.
# The `mkdir -p` ensures the parent directories exist before `chown` if they weren't created by COPY.
RUN mkdir -p /home/appuser/.cache/torch/sentence_transformers && \
    chown -R appuser:appuser /home/appuser/.cache/torch

# Expose the port Streamlit will run on (Cloud Run default is 8080)
EXPOSE 8080

# Healthcheck for container orchestration
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8080/_stcore/health || exit 1

# Command to run the Streamlit application with production-ready flags
CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.headless", "true", "--server.enableCORS", "true", "--server.enableXsrfProtection", "true", "--server.runOnSave", "false", "--server.fileWatcherType", "none"]