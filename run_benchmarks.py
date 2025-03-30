import os
import json
import re
import platform
import psutil
from typing import List, Tuple, TypedDict
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph

# --- LangSmith Tracing Configuration ---
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "planner-optimisation-v1"

# --- Filepath to Prequantised Model (GGUF) ---ƒ
MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--TheBloke--Mistral-7B-Instruct-v0.2-GGUF/snapshots/3a6fbf4a41a1d52e415a4958cde6856d34b2db93/mistral-7b-instruct-v0.2.Q3_K_M.gguf"
)


# --- LangGraph State Definition ---
class PlannerState(TypedDict, total=False):
    task: str
    history: List[Tuple[str, str]]
    plan: str
    golden_plan: str
    subtasks: List[str]
    completed_subtasks: List[str]
    constraint_failures: List[str]
    reasoning: str
    judgement: dict
    system_metrics: dict


# --- LLM Wrapper With Edge Metadata ---
base_llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.2,
    n_ctx=2048,
    n_threads=8,
    n_gpu_layers=35,
    f16_kv=True,
    n_predict=1024,
    verbose=False,
)

llm = base_llm.with_config(
    {
        "metadata": {
            "llm_model_format": "gguf",
            "quantisation": "Q4_K_M",
            "backend": "llama.cpp",
            "ram_usage": "under_8GB",
            "prompt_format": "plain-text",
        },
        "tags": ["edge", "int4", "quantised", "local"],
    }
)

# --- Prompt Templates ---
subtask_prompt = PromptTemplate.from_template(
"""
    You are a task planner assistant. Break the user's high-level request into concrete subtasks.
    
    Original task:
    ""\"
    {task}
    ""\"
    
    Return a bullet point list of subtasks.
"""
)

planner_prompt = PromptTemplate.from_template(
"""
    You are an expert meeting planner. Your job is to select the earliest time slot that:
    - Fits all participants’ constraints
    - Does NOT overlap with existing meetings
    - Respects preferences (like “prefer mornings” or “avoid late meetings”)
    
    You must:
    1. Parse the blocked schedules for each participant
    2. Identify **all non-conflicting slots**
    3. Pick the **earliest valid one**
    
    Respond using only:
    "Here is the proposed time: <Day>, <HH:MM> - <HH:MM>"
"""
)

judger_prompt = PromptTemplate.from_template(
"""
    You are an expert evaluator for calendar scheduling tasks.
    
    Your job is to compare a model-generated meeting plan (called the *prediction*) with a reference plan (called the *golden plan*). Both plans must specify a meeting time in the format:
    
        "<Day>, <HH:MM> - <HH:MM>"
    
    Return only JSON in this format:
    {{
      "score": 1 or 0,
      "reason": "short natural language explanation"
    }}

---

    TASK: {task}
    
    PREDICTION: {prediction}
    GOLDEN: {golden_plan}
    
    Compare both plans.
    
    If:
    - Day matches (case-insensitive)
    - Start and end times match exactly
    
    → score = 1
    
    Else → score = 0
    
    Return only JSON as shown.
"""
)


# --- System Info Helper ---
def get_system_metrics():
    mem = psutil.virtual_memory()
    return {
        "ram_total_gb": round(mem.total / 1e9, 2),
        "ram_used_gb": round(mem.used / 1e9, 2),
        "cpu": platform.processor(),
        "platform": platform.system(),
    }


# --- LangGraph Nodes ---
def subtask_generator(state: PlannerState) -> PlannerState:
    response = (subtask_prompt | llm).invoke({"task": state["task"]})
    subtasks = [
        line.strip("- ") for line in response.strip().split("\n") if line.strip()
    ]
    return {**state, "subtasks": subtasks, "completed_subtasks": []}


def planner_node(state: PlannerState) -> PlannerState:
    result = (planner_prompt | llm).invoke({"task": state["task"]})
    return {**state, "plan": result.strip()}


def judge_node(state: PlannerState) -> PlannerState:
    prediction = state.get("plan")
    golden = state.get("golden_plan")
    if not golden:
        return state

    judge_input = {
        "task": state["task"],
        "prediction": prediction,
        "golden_plan": golden,
    }
    judgement_raw = (judger_prompt | llm).invoke(judge_input)

    # 🔧 Extract first JSON object from messy LLM output
    match = re.search(r"\{.*?\}", judgement_raw, re.DOTALL)
    if match:
        try:
            judgement = json.loads(match.group(0))
        except json.JSONDecodeError as e:
            judgement = {
                "score": 0,
                "reason": f"Failed to parse JSON: {e}",
                "raw": judgement_raw,
            }
    else:
        judgement = {
            "score": 0,
            "reason": "No JSON object found in LLM output.",
            "raw": judgement_raw,
        }

    state["judgement"] = judgement
    return state


def state_tracker(state: PlannerState) -> PlannerState:
    completed = state.get("completed_subtasks", [])
    if "plan" in state and "subtasks" in state:
        for sub in state["subtasks"]:
            if sub.lower() in state["plan"].lower() and sub not in completed:
                completed.append(sub)
    state["completed_subtasks"] = completed
    return state


def log_feedback(state: PlannerState) -> PlannerState:
    print("\n--- PLAN ---\n", state.get("plan"))
    if "judgement" in state:
        print("✅ Evaluation:", state["judgement"])
    if "completed_subtasks" in state:
        print("🧠 Subtasks Completed:", state["completed_subtasks"])
    print("📟 System:", state.get("system_metrics"))
    return state


# --- LangGraph Construction ---
graph = StateGraph(PlannerState)
graph.add_node("subtask_generator", subtask_generator)
graph.add_node("planner", planner_node)
graph.add_node("judge", judge_node)
graph.add_node("state_tracker", state_tracker)
graph.add_node("feedback", log_feedback)

graph.set_entry_point("subtask_generator")
graph.add_edge("subtask_generator", "planner")
graph.add_edge("planner", "state_tracker")
graph.add_edge("state_tracker", "judge")
graph.add_edge("judge", "feedback")
graph.set_finish_point("feedback")
compiled_graph = graph.compile()

# --- Entry Point for Test Execution ---
if __name__ == "__main__":
    # Load calendar scheduling examples from file
    dataset_path = "data/calendar_scheduling_langsmith_ready.jsonl"
    EXAMPLES = []
    with open(dataset_path, "r") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            item = json.loads(line)
            EXAMPLES.append(
                {
                    "task": item["inputs"]["prompt"],
                    "golden_plan": item["outputs"]["golden_plan"],
                }
            )

    total, correct = 0, 0
    for example in EXAMPLES:
        example["system_metrics"] = get_system_metrics()
        final = compiled_graph.invoke(example)
        if "judgement" in final and final["judgement"]["score"] == 1:
            correct += 1
        total += 1
    print(f"\n📊 Final Accuracy: {correct}/{total} = {(correct/total)*100:.2f}%")
