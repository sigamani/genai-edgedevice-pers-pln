# langgraph_planner_pipeline.py

import os
import json
import re
from datetime import datetime, timedelta
from typing import List, Tuple, TypedDict

from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langsmith.client import Client
from langgraph.graph import StateGraph, END

# --- Enable LangSmith tracing via environment variables ---
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "planner-optimisation-v1"
os.environ["LANGCHAIN_PROJECT"] = "planner-optimisation-v1"

# --- Configuration ---
MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--TheBloke--Mistral-7B-Instruct-v0.2-GGUF/snapshots/3a6fbf4a41a1d52e415a4958cde6856d34b2db93/mistral-7b-instruct-v0.2.Q3_K_M.gguf"
)
PROJECT_NAME = "planner-optimisation-v1"
DATASET_PATH = "data/calendar_scheduling_langsmith_ready.jsonl"

# --- LangSmith Setup ---
client = Client()


# --- LangGraph State ---
class PlannerState(TypedDict, total=False):
    task: str
    history: List[Tuple[str, str]]
    plan: str
    run_id: str
    golden_plan: str
    reasoning: str
    valid_slots: List[str]


# --- LLM Setup ---
llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.2,
    n_ctx=2048,
    n_threads=8,
    n_gpu_layers=35,
    f16_kv=True,
    n_predict=768,
    verbose=False,
)

# --- Prompts ---
planner_prompt = PromptTemplate.from_template(
    """
You are an expert at scheduling meetings. Given constraints on participant availability, propose a valid meeting time.
Use this format: Here is the proposed time: <Day>, <HH:MM> - <HH:MM>

TASK: {task}
SOLUTION:
"""
)

reasoning_prompt = PromptTemplate.from_template(
    """
You proposed this meeting time: "{previous_answer}"

However, it may conflict with one or more participants' schedules or preferences.

Here are the constraints again:
{constraints}

Below is a list of all possible valid half-hour slots that do NOT conflict with any participant’s schedule:
{valid_slots}

Please choose the earliest available slot from the list and propose a new meeting time in the format:
"Here is the proposed time: <Day>, <HH:MM> - <HH:MM>"

Explain your reasoning before the final answer.
"""
)


# --- Utilities ---
def extract_meeting_time(text: str) -> str | None:
    match = re.search(r"Here is the proposed time: (.*)", text)
    return match.group(1).strip() if match else None


def parse_blocked_times(constraints: str) -> list[tuple[datetime, datetime]]:
    pattern = r"(\d{1,2}:\d{2})\s*to\s*(\d{1,2}:\d{2})"
    matches = re.findall(pattern, constraints)
    return [
        (datetime.strptime(s, "%H:%M"), datetime.strptime(e, "%H:%M"))
        for s, e in matches
    ]


def get_valid_half_hour_slots(start="09:00", end="17:00", blocked=None):
    if blocked is None:
        blocked = []
    cur = datetime.strptime(start, "%H:%M")
    end = datetime.strptime(end, "%H:%M") - timedelta(minutes=30)
    slots = []
    while cur <= end:
        s, e = cur, cur + timedelta(minutes=30)
        if not any(max(s, b1) < min(e, b2) for b1, b2 in blocked):
            slots.append(f"{s.strftime('%H:%M')} - {e.strftime('%H:%M')}")
        cur += timedelta(minutes=30)
    return slots


# --- LangGraph Nodes ---
def planner_node(state: PlannerState) -> PlannerState:
    result = (planner_prompt | llm).invoke({"task": state["task"]})
    return {**state, "plan": result.strip()}


def reasoner_node(state: PlannerState) -> PlannerState:
    proposed = extract_meeting_time(state["plan"])
    blocked = parse_blocked_times(state["task"])
    valid = get_valid_half_hour_slots(blocked=blocked)
    valid_str = "\n".join(f"- {s}" for s in valid)

    if proposed not in valid:
        result = (reasoning_prompt | llm).invoke(
            {
                "previous_answer": proposed,
                "constraints": state["task"],
                "valid_slots": valid_str,
            }
        )
        state["plan"] = result.strip()
        state["reasoning"] = result
        state["valid_slots"] = valid
    return state


def log_feedback(state: PlannerState) -> PlannerState:
    plan = extract_meeting_time(state["plan"])
    match = plan == state.get("golden_plan")
    client.create_feedback(
        run_id=state.get("run_id", ""),
        key="match_to_golden_plan",
        score=1 if match else 0,
        comment=f"Match to golden plan: {match}",
    )
    return state


# --- LangGraph Build ---
graph = StateGraph(PlannerState)
graph.add_node("planner", planner_node)
graph.add_node("reasoner", reasoner_node)
graph.add_node("feedback", log_feedback)
graph.set_entry_point("planner")
graph.add_edge("planner", "reasoner")
graph.add_edge("reasoner", "feedback")
graph.set_finish_point("feedback")
compiled_graph = graph.compile()

# --- Run on Dataset ---
if __name__ == "__main__":
    with open(DATASET_PATH) as f:
        for line in f:
            entry = json.loads(line)
            task = entry["inputs"]["prompt"]
            golden = entry["outputs"]["golden_plan"]
            print("\n---\nTASK:", task[:80], "...")
            final = compiled_graph.invoke({"task": task, "golden_plan": golden})
            print("PLAN:", final["plan"])
            print("GOLDEN:", golden)
            if "reasoning" in final:
                print("REASONING:\n", final["reasoning"][:300], "...")
