# Base image with Python 3.11
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (ffmpeg + build tools for Whisper, Llama)
RUN apt-get update && \
    apt-get install -y ffmpeg build-essential git curl && \
    apt-get clean

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install -r requirements.txt

# Activate the venv for runtime
ENV PATH="/opt/venv/bin:$PATH"

# Expose Flask port
EXPOSE 8000

# Entry point
CMD ["python", "server.py"]
