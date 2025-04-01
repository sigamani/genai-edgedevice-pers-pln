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
os.environ["LLAMA_LOG_LEVEL"] = "ERROR"

import time, platform, psutil

class BenchmarkLogger:
    def __init__(self, model_name: str = "unknown", n_predict: int = 0):
        self.model_name = model_name
        self.n_predict = n_predict
        self.reset()

    def reset(self):
        self.start_time = None
        self.first_token_time = None
        self.end_time = None
        self.token_count = 0

    def token_callback(self, token: str):
        if self.first_token_time is None:
            self.first_token_time = time.time()
        self.token_count += 1

    def start(self, input_text: str):
        self.reset()
        self.input_token_count = len(input_text.split())  # basic proxy
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    def results(self):
        total_time = self.end_time - self.start_time if self.end_time else None
        ttft = self.first_token_time - self.start_time if self.first_token_time else None
        completion_time = total_time - ttft if total_time and ttft else None
        throughput = self.token_count / total_time if total_time else 0.0
        return {
            "model": self.model_name,
            "n_predict": self.n_predict,
            "tokens_input": self.input_token_count,
            "tokens_output": self.token_count,
            "latency_total_sec": round(total_time, 3),
            "time_to_first_token_sec": round(ttft, 3) if ttft else None,
            "completion_time_sec": round(completion_time, 3) if completion_time else None,
            "throughput_tokens_per_sec": round(throughput, 2),
            "system": {
                "platform": platform.system(),
                "cpu": platform.processor(),
                "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 2),
                "ram_used_gb": round(psutil.virtual_memory().used / 1e9, 2),
            }
        }

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
    model_path="models/mistral-7b-instruct-v0.2.Q3_K_M.gguf",
    temperature=0.2,
    n_ctx=1024,
    n_threads=os.cpu_count(),
    n_gpu_layers=64,
    f16_kv=True,
    n_predict=128,
    n_batch=64,
    llama_log_level="error",
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

    # Extract first JSON object from messy LLM output
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
        print("Evaluation:", state["judgement"])
    if "completed_subtasks" in state:
        print("🧠 Subtasks Completed:", state["completed_subtasks"])
    print("System:", state.get("system_metrics"))
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
 #   import wandb
 #   wandb.init(project="agentic-planner-8b", name="run_benchmarks_eval", config={"model": "mistral-7b-instruct.Q3_K_M", "n_predict": 124})

    # Load calendar scheduling examples from file
    dataset_path = "data/calendar_scheduling_langsmith_ready.jsonl"
    EXAMPLES = []
    with open(dataset_path, "r") as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            task_text = item.get("inputs", {}).get("prompt", "")
            golden_plan_text = item.get("outputs", {}).get("golden_plan", "")
            EXAMPLES.append({
                "task": task_text.strip(),
                "golden_plan": golden_plan_text.strip(),
            })

    total, correct = 0, 0
    from langchain_core.callbacks.base import BaseCallbackHandler

    class TokenCallbackHandler(BaseCallbackHandler):
        def __init__(self, callback_fn):
            self.callback_fn = callback_fn

        def on_llm_new_token(self, token, **kwargs):
            self.callback_fn(token)

    for example in EXAMPLES:
        example["system_metrics"] = get_system_metrics()
        final = compiled_graph.invoke(example)
        logger = BenchmarkLogger(model_name="mistral-7b-instruct.Q3_K_M", n_predict=124)
        logger.start(example["task"])

        # Inject callback to track tokens
        token_handler = TokenCallbackHandler(logger.token_callback)
        compiled_graph.invoke(example, config={"callbacks": [token_handler]})

        logger.stop()
        print(logger.results())  # or add to a final results list
     #   wandb.log(logger.results())

        if "judgement" in final and final["judgement"]["score"] == 1:
            correct += 1
        total += 1
    print(f"\n Final Accuracy: {correct}/{total} = {(correct/total)*100:.2f}%")
 #   wandb.log({"final_accuracy": correct / total})
#    wandb.finish()

# --- End of run_benchmarks.py ---
