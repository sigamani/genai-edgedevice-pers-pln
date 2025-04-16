# Strategy for an Edge-Optimized AI Planner Model
## Introduction

Planning tasks like trip itineraries and meeting scheduling pose significant challenges for LLMs. The [NATURAL PLAN benchmark](https://arxiv.org/abs/2406.04520) tests LLMs on realistic planning problems, including trip planning (budget/flight constraints), meeting planning (across friends and locations), and calendar scheduling. These tasks require the model to generate valid plans from full context inputs. Even SOTA models like GPT-4 (31% solve rate) and Gemini 1.5 (35%) underperform, highlighting the difficulty of constraint-based reasoning.

This document outlines our strategy for building an efficient planner model (≤8B params), trained on text-based planning tasks and deployable on edge devices with <30s latency per query. It covers:

- **Model selection and architecture**  
- **Training data and fine-tuning strategies**  
- **Evaluation metrics and benchmarks**  
- **Deployment and quantization for edge**  
- **Integration into assistant workflows**

Our focus is high planning accuracy, minimal latency, and offline privacy-preserving operation.

---

## Model Selection and Architecture

- **Chosen model**: Meta's [LLaMA 3.1 8B](https://ai.meta.com/llama/) for its balance between reasoning capacity and edge deployability.
- **Why 8B?**: Large enough for multi-step reasoning, small enough to run on mobile (<4 GB with quantization).
- **Architecture tweaks**:
  - Retain rotary embeddings, RMSNorm.
  - Extend context to 4K+ tokens via interpolation if needed.
  - Instruction tuning with variants like LLaMA-3.1-8B-Instruct.
  - Add formatting tokens (e.g. `[CALENDAR]`) to separate context parts.
  - Optionally use grouped-query attention for memory efficiency (with caution).

**Trade-off**: Larger models (13B+) offer higher accuracy but are not deployable on common edge hardware. 7B strikes the right balance.

---

## Training Approach

### 1. **Data Collection**
- Use NATURAL PLAN for evaluation and inspiration (~3.6K examples).
- Generate synthetic planning examples (trip plans, meeting slots) using rules.
- Use GPT-4 to create & verify high-quality solutions (distillation).
- Mix synthetic and teacher-model data to balance realism and volume.

### 2. **Supervised Fine-Tuning (SFT) (see other branch) **
- Format inputs as `prompt` (with constraints) and `response` (step-by-step reasoning + final plan).
- Use [LoRA](https://arxiv.org/abs/2106.09685) or [QLoRA](https://arxiv.org/abs/2305.14314) for efficient tuning.
- Train from easy to hard (curriculum learning).
- Validate using held-out NATURAL PLAN examples.

### 3. **Enhancements**
- Add reasoning supervision (chain-of-thought).
- Distill from larger models (not enough time to implement).
- Use reward modelling or RLHF if performance plateaus (not enough time to implement).

---

## Evaluation Protocol (see run_benchmarks.py)

- **Accuracy**: Solve rate on NATURAL PLAN tasks. Use validator scripts to check constraints.
- **Latency**: Target <30s end-to-end on phones or Jetson boards. Measure token/sec and total generation time.
- **Memory**: Ensure model + context fits in 4GB RAM with 4-bit quantization.
- **Output Quality**: Human ratings for fluency, logic, structure.
- **Robustness**: Graceful handling of unsatisfiable constraints.
- **Integration**: Consistent, parsable format for use in assistants.

---
## Example Prompt:

```
Task: Schedule a 30-minute meeting for Julie, Betty, Kayla, Heather, and Keith on Monday.
Work hours: 09:00–17:00
Schedules: {...}
Preferences: {...}
```

## Output
```
{
  "reasoning_steps": [
    "Check availabilities...",
    "Apply preferences..."
  ],
  "final_answer": "Monday 11:00–11:30"
}
```


## Optimization for Edge Deployment
- **Quantization**: Use 4-bit formats like `q4f16_1` (via [MLC-LLM](https://mlc.ai/mlc-llm/) or GPTQ).
- **Runtime**: Compile with MLC for target (Metal/Vulkan/CUDA/TVM).

  
## Conclusion
This strategy delivers a 7B AI planner with high accuracy, fast latency, and reliable integration into assistants — all running on-device. It leverages instruction tuning, data distillation, LoRA fine-tuning, and MLC quantization to build a robust planner that performs nearly on par with GPT-4 for domain-specific planning.

---
## Build Backend

```bash
mlc_llm build --model path/to/llama-2-7b-planner \
              --quantization q4f16_1 \
              --max-seq-len 4096 \
              --target iphone
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

## Citations
	1.	NATURAL PLAN: https://arxiv.org/abs/2406.04520
	2.	LLaMA 2: https://ai.meta.com/llama/
	3.	LoRA: https://arxiv.org/abs/2106.09685
	4.	QLoRA: https://arxiv.org/abs/2305.14314
	5.	MLC-LLM: https://mlc.ai/mlc-llm/
	6.	Alpaca: https://crfm.stanford.edu/2023/03/13/alpaca.html
	7.	FlashAttention: https://arxiv.org/abs/2205.14135
