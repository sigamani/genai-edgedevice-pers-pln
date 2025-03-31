<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">Edge-Deployable AI Planner</h3>
  <p align="center">
    Constraint-aware AI planner using a quantised 8B LLM. Built for CPU and mobile devices with llama.cpp and mlc-llm.
    <br />
    <a href="https://github.com/your-username/ai-planner-on-edge"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="#demo">View Demo</a>
    ·
    <a href="https://github.com/your-username/ai-planner-on-edge/issues">Report Bug</a>
    ·
    <a href="https://github.com/your-username/ai-planner-on-edge/issues">Request Feature</a>
</p>

---

## 🧠 About The Project

This project demonstrates an edge-compatible AI planner capable of handling structured tasks like travel and meeting planning, with budget, time, and weather constraints.

Built using:
- A quantised Mistral 7B or LLaMA2-7B LLM (int4)
- `llama.cpp` for CPU inference
- `mlc-llm` for mobile/WebGPU deployment
- Agentic planning logic with constraint validation
- GitHub Actions CI across platforms

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- `cmake`, `g++`, `wget` (for building llama.cpp)
- Docker (optional)
- GPU (optional for training/fine-tuning)

### Installation

```bash
git clone https://github.com/your-username/agentic-planner-8b.git
cd agentic-planner-8b
pip install -r requirements.txt
```

### Run Planner with Mocked LLM

```bash
python run_planner.py --task "I’d like to visit Japan in May 2025 to see the cherry blossoms, with a total budget of $2,000. Can you help me plan the trip, including recommended destinations, travel tips, and an itinerary within budget?" --backend openai
```

### Run Benchmarks

```bash
python run_benchmarks.py
```

### Run with llama.cpp

```bash
cd llama.cpp
make LLAMA_OPENBLAS=1
./main -m ../models/llama-8b.gguf -p "Plan a trip to Europe for two weeks under $3000"
```

---

## 🧪 Benchmarks

This planner is benchmarked using prompts adapted from [arXiv:2406.04520](https://arxiv.org/pdf/2406.04520). See `benchmarks/` for test cases and evaluation metrics.

---

## 📦 CI/CD

GitHub Actions runs tests on:
- Ubuntu, macOS, Windows
- llama.cpp inference validation
- Planning logic and constraint satisfaction

---

## 📱 Edge Deployment

Models are deployable using:
- `llama.cpp` on CPU-only systems (Raspberry Pi, laptops)
- `mlc-llm` for GPU-accelerated inference on Android, iOS, and WebGPU

---

## 📄 License

Distributed under the Apache 2.0 License.
