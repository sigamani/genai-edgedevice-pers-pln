FROM python:3.10-slim

WORKDIR /app

# Copy source code from the host context (repo is already checked out)
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y git cmake build-essential wget && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt
