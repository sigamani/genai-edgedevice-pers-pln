# Edge-Deployable AI Planner

**📌 Purpose of This Branch**

This branch serves as a separate environment to add core functionality missing in `main` (post-project submission):

- 🔧 Fixing the broken CI/CD pipeline  
- ✅ Adding unit tests for core functionality  
- 🧹 Refactoring the codebase for improved modularity  
- 🐳 Building and containerising this application with Docker  
- 🚀 Integrating deployment via MLC-LLM  
- 📊 Extending and automating benchmarking processes  
- 🧠 Enhancing the agentic workflow by adding a routing step post-input prompt  

These changes aim to create a cleaner, more maintainable, and production-ready codebase.

---

## 🐳 Run the Planner from Docker Hub

You can run the quantised AI planner agent directly using our pre-built Docker image. This image simulates a local, edge-deployable AI system using the [Mistral 7B Instruct (Q3_K_M)](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q3_K_M.gguf) model, optimised for CPU inference via `llama.cpp`.

### 🔧 Requirements

- Docker installed on your machine
- At least ~4GB free disk space
- macOS (Apple Silicon) or Linux recommended

---

### 📦 Step 1: Pull the Container

```bash
docker pull michaelsigamani/agentic-planner-8b:latest
```

---

### ▶️ Step 2: Run Calendar Scheduling Benchmarks

```bash
docker run --rm -it michaelsigamani/agentic-planner-8b:latest python run_cal_benchmarks.py
```

This command will:

- Run the **Calendar Scheduling** benchmark from Google DeepMind’s [NaturalPlan](https://arxiv.org/pdf/2406.04520) dataset
- Use **in-context learning** with a **4-bit quantised model**
- Log evaluation metrics to **Weights & Biases** and traces to **LangSmith**

---

### 🔁 What's Next?

You can:

- Modify `run_cal_benchmarks.py` to target other subtasks like trip planning
- Mount your own `gguf` models to `/models` and adjust the `LlamaCpp` config
- Extend LangGraph logic for smarter agent routing and retries
- Integrate with `mlc-llm` for mobile or GPU-accelerated inference

---
