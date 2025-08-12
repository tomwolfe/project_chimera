# Dockerfile

# Use a lightweight Python image as the base
FROM python:3.11-slim AS builder
WORKDIR /app

# Install production dependencies
COPY requirements-prod.txt .
# Ensure all necessary packages, including streamlit, google-genai, etc., are installed.
RUN pip install --no-cache-dir -r requirements-prod.txt

# Pre-download sentence-transformers model to avoid runtime download issues
# This ensures the model is available at runtime and avoids network issues or delays
# The model 'all-MiniLM-L6-v2' is used in src/context/context_analyzer.py
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Use a non-root user for security
RUN useradd -m -u 1000 appuser
USER appuser

# Copy application code
COPY . .

# Expose the port Streamlit will run on (Cloud Run default is 8080)
EXPOSE 8080

# Healthcheck for container orchestration
# This checks Streamlit's internal health endpoint, which is essential for orchestrators
# like Kubernetes or Cloud Run to determine if the container is ready.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8080/_stcore/health || exit 1

# Command to run the Streamlit application with production-ready flags
# --server.headless: Essential for running in containers without a browser.
# --server.port: Specifies the port the server listens on.
# --server.enableCORS=true: IMPORTANT! Enables Cross-Origin Resource Sharing for Streamlit's internal communication.
# --server.enableXsrfProtection=true: Enables Cross-Site Request Forgery protection.
# --server.runOnSave=false: Prevents Streamlit from automatically reloading on file changes in production.
# --server.fileWatcherType=none: Disables file watching entirely in production for performance.
CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.headless", "true", "--server.enableCORS", "true", "--server.enableXsrfProtection", "true", "--server.runOnSave", "false", "--server.fileWatcherType", "none"]