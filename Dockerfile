# Use official slim image for Python 3.12 with AMD64 architecture
FROM --platform=linux/amd64 python:3.12-slim

# Set working directory
WORKDIR /app

# Copy all local files into the container
COPY . /app

# Pre-install essential build packages and dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables to ensure offline usage
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose input/output structure
VOLUME ["/app/input", "/app/output"]

# Run the script on container start
ENTRYPOINT ["python", "main.py"]
