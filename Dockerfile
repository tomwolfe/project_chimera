# Use a lightweight Python image as the base
FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's build cache
COPY requirements.txt .
# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application files into the container
# This includes app.py, core.py, llm_provider.py, personas.yaml, main.py, LICENSE, README.md
COPY . .

# Cloud Run services listen on the port defined by the PORT environment variable,
# which is typically 8080. Streamlit must be configured to use this port.
# The EXPOSE instruction is for documentation and not strictly required by Cloud Run.
EXPOSE 8080

# Command to run the Streamlit application.
# Using the shell form of CMD to allow $PORT environment variable expansion.
CMD streamlit run app.py --server.port $PORT --server.address 0.0.0.0