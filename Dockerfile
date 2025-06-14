#Dockerfile-backend for FastAPI application(backend)

FROM python:3.11-slim-bullseye AS fastapi-build

WORKDIR /app
# Copy the requirements file first to leverage Docker cache

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
