import json
import os
import concurrent.futures
import argparse
from run_planner import workflow
from langsmith import traceable as ls_traceable
from langsmith import trace
import wandb
from planner_model_runners import llama_cpp_runner

BENCHMARKS_DIR = os.path.join(os.path.dirname(__file__), "benchmarks")
JSONL_FILES = [
    "calendar_scheduling.jsonl",
    "trip_planning.jsonl",
    "meeting_planning.jsonl",
]

# Initialise Weights & Biases logging
wandb.init(
    project="agentic-planner-benchmark", name="benchmark-run", job_type="benchmark"
)


@ls_traceable(name="benchmark-run")
def run_benchmark(backend, model_path=None):
    results = []

    def process_line(line, filename):
        prompt_data = json.loads(line)
        task_input = prompt_data.get("input")
        constraints = prompt_data.get("constraints", {})
        expected_properties = prompt_data.get("expected_properties", {})

        input_state = {
            "task": task_input,
            "constraints": constraints,
            "backend": backend,
            "model_path": model_path,
        }

        with trace(name="benchmark-single-run", tags=[f"{filename}", backend]) as run:
            if backend == "llama.cpp":
                result = llama_cpp_runner(task_input, model_path)
            else:
                result = workflow.invoke(input_state)
            run.output = result  # optional but recommended
            run.add_tags([f"validation_passed:{result.get('validation_passed', False)}"])

            result_entry = {
                "task": task_input,
                "constraints": constraints,
                "plan": result.get("plan"),
                "validation_passed": result.get("validation_passed"),
                "tools_used": result.get("tools_used"),
                "expected_properties": expected_properties,
            }

            wandb.log(
                {
                    "input_text": task_input,
                    "plan_text": result.get("plan", ""),
                    "validation_passed": int(result.get("validation_passed", False)),
                    "tools_used_str": ", ".join(result.get("tools_used", []))
                    if result.get("tools_used")
                    else "",
                    "num_tools_used": len(result.get("tools_used", [])),
                    "filename": filename,
                }
            )

            return result_entry

    for filename in JSONL_FILES:
        file_path = os.path.join(BENCHMARKS_DIR, filename)
        print(f"🔍 Running benchmarks from {file_path}")

        with open(file_path, "r") as f:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for line in f:
                    futures.append(executor.submit(process_line, line, filename))
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())

    print(f"✅ Evaluated {len(results)} prompts across all benchmarks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmarks for agentic planner")
    parser.add_argument(
        "--backend",
        type=str,
        default="llama.cpp",
        help="Model backend to use (openai, llama.cpp, mlc-llm)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=False,
        help="Path to local GGUF model for llama.cpp backend",
    )
    args = parser.parse_args()

    run_benchmark(args.backend, args.model_path)
    wandb.finish()
