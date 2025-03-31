FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone your repo directly
RUN git clone https://github.com/sigamani/agentic-planner-8b.git . --depth=1

RUN pip install --no-cache-dir -r requirements.txt

# Build llama.cpp (optional)
RUN git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && \
    make LLAMA_OPENBLAS=1

ENTRYPOINT ["python", "run_graph_planner.py"]
