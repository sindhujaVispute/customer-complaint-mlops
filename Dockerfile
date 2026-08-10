# Dockerfile for Customer Complaint Classification MLOps Project

# Use Python 3.10 slim image for smaller size
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    TRANSFORMERS_CACHE=/app/cache/transformers \
    HF_DATASETS_CACHE=/app/cache/datasets \
    MLFLOW_TRACKING_URI=/app/mlruns

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p /app/artifacts /app/cache/transformers /app/cache/datasets /app/mlruns

# Expose port for potential API
EXPOSE 5000 8000

# Set entrypoint
ENTRYPOINT ["python"]

# Default command (can be overridden)
CMD ["predict.py", "--interactive"]