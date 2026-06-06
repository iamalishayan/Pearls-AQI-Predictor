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

# Set API URL for Streamlit to find FastAPI in the same container (internal network)
ENV API_URL=http://localhost:8000

# Start both services — API in background (internal), Streamlit in foreground (public on Render's $PORT)
CMD uvicorn src.app.api:app --host 0.0.0.0 --port 8000 & \
    streamlit run src/app/dashboard.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true
