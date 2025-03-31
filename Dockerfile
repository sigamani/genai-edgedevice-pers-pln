FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone your repo
RUN git clone https://github.com/sigamani/agentic-planner-8b.git . --depth=1

# Install Python requirements
RUN pip install --no-cache-dir -r requirements.txt

# Clone llama.cpp and build with Metal (macOS target)
RUN git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && \
    make LLAMA_METAL=1

# Download model from Hugging Face (ensure token is passed)
ARG HF_API_KEY
ARG MODEL_NAME=TheBloke/Mistral-7B-Instruct-v0.2-GGUF
ARG MODEL_FILE=mistral-7b-instruct-v0.2.Q3_K_M.gguf

RUN mkdir -p /app/models && \
    curl -L -H "Authorization: Bearer ${HF_API_KEY}" \
    https://huggingface.co/${MODEL_NAME}/resolve/main/${MODEL_FILE} \
    -o /app/models/${MODEL_FILE}

# Set benchmark input
COPY benchmarks/calendar_scheduling_langsmith_ready.jsonl /app/benchmark_input.jsonl

# Default run
ENTRYPOINT ["python", "run_cal_benchmarks.py", "--input", "benchmark_input.jsonl"]
