import json
import os
from run_planner import workflow
from langsmith import traceable as ls_traceable
from langsmith.run_helpers import trace
import wandb

BENCHMARKS_DIR = os.path.join(os.path.dirname(__file__), "benchmarks")
JSONL_FILES = [
    "calendar_scheduling.jsonl",
 #   "trip_planning.jsonl",
 #   "meeting_planning.jsonl"
    ]

# Initialise Weights & Biases logging
wandb.init(project="agentic-planner-benchmark", name="benchmark-run", job_type="benchmark")


@ls_traceable(name="BenchmarkLoop")
def run_benchmark():
    results = []

    for filename in JSONL_FILES:
        file_path = os.path.join(BENCHMARKS_DIR, filename)
        print(f"Running benchmarks from {file_path}")

        with open(file_path, "r") as f:
            for line in f:
                prompt_data = json.loads(line)
                task_input = prompt_data.get("input")
                constraints = prompt_data.get("constraints", {})
                expected_properties = prompt_data.get("expected_properties", {})

                input_state = {
                    "task": task_input,
                    "constraints": constraints,
                    "backend": "openai"  # or "llama.cpp", etc.
                }

                result = workflow.invoke(input_state)

                with trace(name="benchmark-single-run") as run:
                    run.add_tags([f"validation_passed:{result.get('validation_passed', False)}"])

                    result_entry = {
                        "task": task_input,
                        "constraints": constraints,
                        "plan": result.get("plan"),
                        "validation_passed": result.get("validation_passed"),
                        "tools_used": result.get("tools_used"),
                        "expected_properties": expected_properties
                    }

                    # Log to Weights & Biases
                    wandb.log({
                        "input_text": task_input,
                        "plan_text": result.get("plan", ""),
                        "validation_passed": int(result.get("validation_passed", False)),
                        "tools_used_str": ", ".join(result.get("tools_used", [])) if result.get("tools_used") else "",
                        "num_tools_used": len(result.get("tools_used", [])),
                        "filename": filename
                    })

                    results.append(result_entry)

    print(f"Evaluated {len(results)} prompts across all benchmarks.")


if __name__ == "__main__":
    run_benchmark()
    wandb.finish()
