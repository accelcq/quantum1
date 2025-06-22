#Dockerfile-backend for FastAPI application(backend)

FROM python:3.11-slim

WORKDIR /app

# Copy requirements.txt from app/ to /app in the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all backend source code into the container's /app directory
COPY . .

# Start the FastAPI app using uvicorn
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
