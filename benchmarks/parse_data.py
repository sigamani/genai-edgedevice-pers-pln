import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NATURAL_PLAN_PATH = os.path.join(BASE_DIR, "natural-plan/data/calendar_scheduling.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "calendar_scheduling.jsonl")


def convert_naturalplan_to_prompts():
    with open(NATURAL_PLAN_PATH, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            raw = f.read()
            data = [
                json.loads(json.loads(line))
                for line in raw.splitlines()
                if line.strip()
            ]
    prompts = []
    for i, (example_id, item) in enumerate(data.items()):
        task_input = item.get("prompt_0shot", "")
        constraints = {
            "cities": item.get("cities", ""),
            "durations": item.get("durations", ""),
        }
        expected_properties = {"golden_plan": item.get("golden_plan", "")}

        prompt_entry = {
            "id": f"{example_id}",
            "task_type": "trip",
            "constraints": constraints,
            "input": task_input,
            "expected_properties": expected_properties,
        }
        prompts.append(prompt_entry)

    with open(OUTPUT_PATH, "w") as f:
        for entry in prompts:
            f.write(json.dumps(entry) + "\n")

    print(f"✅ Converted {len(prompts)} prompts and saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    convert_naturalplan_to_prompts()
