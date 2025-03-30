from typing import TypedDict
import json
import re
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp

# Initialise your LLM (adjust path and params as needed)
llm = LlamaCpp(
    model_path="/Users/michaelsigamani/.cache/huggingface/hub/models--TheBloke--Mistral-7B-Instruct-v0.2-GGUF/snapshots/3a6fbf4a41a1d52e415a4958cde6856d34b2db93/mistral-7b-instruct-v0.2.Q3_K_M.gguf",
    n_ctx=2048,
    n_batch=512,
    temperature=0.1,
    model_kwargs={"n_predict": 128}
)

class PlannerState(TypedDict, total=False):
    task: str
    plan: str
    golden_plan: str
    judgement: dict

def judge_node(state: PlannerState) -> PlannerState:
    prediction = state.get("plan")
    golden = state.get("golden_plan")
    task = state.get("task", "")

    if not golden:
        return state

    # Task-type-specific judging logic
    if "schedule" in task.lower() or "meeting" in task.lower():
        judge_prompt = PromptTemplate.from_template(
            """
You are an expert evaluator for calendar scheduling tasks.

Your job is to compare a model-generated meeting plan (called the *prediction*) with a reference plan (called the *golden plan*). Both plans must specify a meeting time in this exact format:

    "<Day>, <HH:MM> - <HH:MM>"

Always output **only** a JSON object as your first line. For example:

{
  "score": 1,
  "reason": "The plans match exactly."
}

or

{
  "score": 0,
  "reason": "The plans do not match."
}

TASK: {task}

PREDICTION: {prediction}

GOLDEN: {golden_plan}

Respond with only JSON and no extra text.
"""
        )
    else:
        judge_prompt = PromptTemplate.from_template(
            """
You are an expert evaluator for structured planning tasks, such as trip itineraries or project timelines.

Compare the *prediction* and *golden plan*. Score 1 if:
- They include the same key locations or goals
- The sequence or timeline is logically consistent with the golden plan
- The constraints (like rest periods or specific stops) are respected

Score 0 if:
- Locations are missing
- The day-to-day structure or city ordering is wrong
- Important constraints are ignored

TASK: {task}

PREDICTION: {prediction}

GOLDEN PLAN: {golden_plan}

Respond with only JSON and no extra text.
"""
        )

    judge_input = {
        "task": task,
        "prediction": prediction,
        "golden_plan": golden,
    }

    judgement_raw = (judge_prompt | llm).invoke(judge_input)
    judgement_raw = judgement_raw.strip().strip("```json").strip("```").strip()

    # JSON extraction
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