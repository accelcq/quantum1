#Dockerfile-backend for FastAPI application(backend)

# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory to root so 'app' becomes a proper Python module
WORKDIR /

# Copy requirements and install dependencies
#COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy all backend source code into the container's /app directory
COPY . /app

# Set Python path so 'app' module can be imported
ENV PYTHONPATH=/

# Start the FastAPI app using uvicorn from the app directory
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
