FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone repo directly
RUN git clone https://<USERNAME>:<TOKEN>@github.com/sigamani/agentic-planner-8b.git . --depth=1

RUN pip install --no-cache-dir -r requirements.txt

# Build llama.cpp (optional)
RUN git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && \
    make LLAMA_OPENBLAS=1

# Download quantised Mistral GGUF model (e.g., from TheBloke)
ARG HF_API_KEY
ARG MODEL_NAME=TheBloke/Mistral-7B-Instruct-v0.2-GGUF
ARG MODEL_FILE=mistral-7b-instruct-v0.2.Q3_K_M.gguf

RUN mkdir -p /app/models && \
    curl -L -H "Authorization: Bearer ${HF_API_KEY}" \
    https://huggingface.co/${MODEL_NAME}/resolve/main/${MODEL_FILE} \
    -o /app/models/${MODEL_FILE}

# Expose the planner runner as default
ENTRYPOINT ["python", "run_graph_planner.py"]
