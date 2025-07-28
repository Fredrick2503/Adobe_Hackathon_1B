FROM python:3.10-slim

# Set environment variable to avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including libgomp and other essential libs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    build-essential \
    gcc \
    g++ \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirement.txt

# Run main.py
CMD ["python", "main.py"]
