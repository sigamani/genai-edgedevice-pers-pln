FROM python:3.10-slim

WORKDIR /app

# ------------------------
# System dependencies
# ------------------------
RUN apt-get update && apt-get install -y \
    git cmake build-essential wget curl \
    && rm -rf /var/lib/apt/lists/*

# ------------------------
# Install Python deps
# ------------------------
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ------------------------
# Clone and build llama.cpp
# ------------------------
RUN git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && \
    cmake . -DLLAMA_BLAS=ON -DLLAMA_NATIVE=ON && \
    make -j$(nproc)

# ------------------------
# Copy your app
# ------------------------
COPY . .

# Set llama binary path as env var
ENV LLAMA_BINARY=/app/llama.cpp/main

CMD ["python", "run_cal_benchmarks.py"]
