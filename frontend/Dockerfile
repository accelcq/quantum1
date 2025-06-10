# frontend/Dockerfile for building and serving a React application with Nginx (frontend)
# Stage 1: Build the React app
FROM node:18 as build
WORKDIR /app
COPY package*.json ./
COPY .env .env

RUN npm install
COPY . .
RUN npm run build

# Stage 2: Serve with Nginx
FROM nginx:stable-alpine
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
# Stage 3: Build and run the FastAPI app , it is taken care of in the backend Dockerfile in the root directory