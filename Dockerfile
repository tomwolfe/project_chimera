# Use a lightweight Python image as the base
FROM python:3.9-slim-buster AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Use a non-root user for security
RUN useradd -m -u 1000 appuser
USER appuser

# Copy application code
COPY . .

# Expose the port Streamlit will run on (Cloud Run default is 8080)
EXPOSE 8080

# Healthcheck for container orchestration
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8080/_stcore/health || exit 1

# Command to run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.headless", "true"]