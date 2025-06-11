# frontend/Dockerfile for building and serving a React application with Nginx (frontend)
# Stage 1: Build the React app
FROM node:18-slim AS build
WORKDIR /app
COPY package*.json ./

# Copy the rest of the application code
# Copy the .env file to the working directory
COPY . .

# Build the React application
RUN npm install
RUN npm run build

# Stage 2: Serve with Nginx
FROM nginx:stable-alpine
RUN apk update && apk upgrade && rm -rf /var/cache/apk/*
RUN rm /etc/nginx/conf.d/default.conf
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
# Stage 3: Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost/ || exit 1
