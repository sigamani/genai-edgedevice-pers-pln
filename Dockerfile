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
# Disable f16 optimisations to avoid NEON inline errors
ENV CMAKE_ARGS="-DLLAMA_F16=OFF -DLLAMA_NATIVE=OFF"

RUN pip install --upgrade pip && pip install -r requirements.txt && pip install llama-cpp-python --no-cache-dir
# # ------------------------
# # Clone and build llama.cpp
# # ------------------------
# RUN git clone https://github.com/ggerganov/llama.cpp.git && \
#     cd llama.cpp && \
#     cmake . -DLLAMA_NATIVE=OFF -DLLAMA_BLAS=OFF -DLLAMA_ACCEL=OFF -DCMAKE_BUILD_TYPE=Release && \
#     make -j$(nproc)

# ------------------------
# Copy your app
# ------------------------
COPY . .

# Set llama binary path as env var
ENV LLAMA_BINARY=/app/llama.cpp/build/bin/llama-run

CMD ["python", "run_cal_benchmarks.py"]
