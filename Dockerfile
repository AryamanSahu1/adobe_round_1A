# Use a CPU-compatible base image for AMD64
FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Ensure model is loaded locally from ./hf_cache or ./model
ENV TRANSFORMERS_CACHE=/app/hf_cache

# Default command to run inference
CMD ["python", "main.py"]
