# syntax=docker/dockerfile:1

# ---------- Stage 1: build the SPA ----------
FROM node:22-alpine AS build
# Testing Lab is excluded from production builds by default.
ARG VITE_ENABLE_LAB=false
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN echo "VITE_ENABLE_LAB=${VITE_ENABLE_LAB}" > .env.production \
    && npm run build

# ---------- Stage 2: serve static files + reverse-proxy the API ----------
FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
