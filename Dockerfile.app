# Use official Python runtime as a parent image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for compilation of some ML packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source code
COPY src/ /app/src/
COPY .env /app/.env

# Default command (can be overridden in docker-compose)
CMD ["python", "-m", "src.app.api"]
