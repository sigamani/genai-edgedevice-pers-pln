# Edge-Deployable AI Planner

**Constraint-aware AI planner using a quantised 8B LLM.**  
Built for CPU and mobile devices using `llama.cpp`.

---

## Project Overview

This project showcases an **edge-compatible AI planning system** capable of handling structured tasks like:

- Trip planning (e.g., multi-city itineraries with budget/time limits)
- Meeting scheduling
- Constraint management (e.g., budget, time, weather)

### Key Features
- Lightweight Mistral 7B (int4 quantised)
- Optimised for `llama.cpp` (CPU)
- Agentic planning loop with constraint validation
- Built-in benchmarking and evaluation with LangSmith and W&B

---

##  Getting Started

### Prerequisites
- Python 3.10+
- `cmake`, `g++`, `wget` (to build `llama.cpp`)
- Docker (optional)

### Installation

```bash
git clone https://github.com/your-username/agentic-planner-8b.git
cd agentic-planner-8b
pip install -r requirements.txt
```

---

## Benchmarks & Evaluation

We benchmark on structured planning tasks using data inspired by [arXiv:2406.04520](https://arxiv.org/pdf/2406.04520).

Run the benchmark suite:
```bash
python run_benchmarks.py --backend llama.cpp
```

Results are logged to:
- [LangSmith](https://smith.langchain.com/public/21b06a5d-4661-4594-874b-86cf733c142b/r)
- [Weights & Biases](https://wandb.ai/michael-sigamani-oxalatech/agentic-planner-8b)

---

## How to Use

### Run the Planner (OpenAI Backend)

```bash
python run_planner.py --task "Plan a budget Europe trip in July" --backend openai
```

### Run the Planner (llama.cpp)

```bash
cd llama.cpp
cmake -B build
cmake --build build
./build/bin/llama-cli -m ./models/llama-8b.Q4_K_M.gguf -p "Plan a trip to Europe for two weeks under $3000"
```

---

## Continuous Integration

CI via GitHub Actions:
- Multi-platform (Ubuntu, macOS, Windows)
- LLM inference tests (`llama.cpp`)
- Planning loop and constraint coverage

---

## Edge Compatibility

Deploy this agent to:
- Laptops (CPU-only)
- Raspberry Pi / Jetson Nano
