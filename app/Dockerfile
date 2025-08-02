#Dockerfile-backend for FastAPI application(backend)

# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/data/predictions /app/logs

# Expose port
EXPOSE 8080

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the application (using the working CMD)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
