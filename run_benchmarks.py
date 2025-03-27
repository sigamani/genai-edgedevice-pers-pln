import json
import wandb
from planner.plan import plan
from planner.mock_runner import mock_model_runner

# --- Simple evaluation logic ---
def evaluate(output: str, expectations: dict) -> dict:
    results = {}
    for key, val in expectations.items():
        results[key] = str(val).lower() in output.lower()
    return results

# --- Main benchmark runner with W&B ---
def main():
    wandb.init(project="agentic-planner-8b", name="benchmark-run", config={"backend": "mock"})
    results = []

    with open("benchmarks/planner_prompts.jsonl") as f:
        for line in f:
            task = json.loads(line)
            output = plan(task["task_type"], task["constraints"], model_runner=mock_model_runner)
            evaluation = evaluate(output, task["expected_properties"])
            score = sum(evaluation.values()) / len(evaluation)

            # Log to wandb
            wandb.log({
                "task_id": task["id"],
                "task_type": task["task_type"],
                "input": task["input"],
                "output": output,
                "score": score,
                **{f"constraint_success/{k}": v for k, v in evaluation.items()}
            })

            results.append({
                "id": task["id"],
                "input": task["input"],
                "output": output,
                "evaluation": evaluation,
                "score": score
            })

    print("\n--- Evaluation Results ---")
    for r in results:
        print(f"{r['id']} — Score: {r['score']*100:.1f}% — {r['evaluation']}")

if __name__ == "__main__":
    main()
