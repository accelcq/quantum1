#Dockerfile
# Stage 1: Build React App
FROM node:18 as build
WORKDIR /app
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ .
RUN npm run build

# Stage 2: Serve with Nginx
FROM nginx:stable-alpine
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
# Stage 3: Build and run the FastAPI app
FROM python:3.11-slim-bullseye AS fastapi-build

WORKDIR /app

# Upgrade pip and system packages to reduce vulnerabilities, then clean up
RUN apt-get update && \
	apt-get dist-upgrade -y && \
	apt-get install --only-upgrade -y python3 && \
	pip install --upgrade pip && \
	apt-get autoremove -y && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
