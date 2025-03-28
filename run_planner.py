from langgraph.graph import StateGraph, END
from langsmith import traceable as ls_traceable
from planner_model_runners import planner_model_runner
from typing import TypedDict, Optional, List, Any
import argparse

# --- Define compatible ToolCallingState ---
class ToolCallingState(TypedDict, total=False):
    task: str
    constraints: dict
    plan: Optional[str]
    validation_passed: Optional[bool]
    tools_used: Optional[List[str]]
    messages: Optional[List[Any]]
    next_steps: Optional[List[Any]]
    step_results: Optional[List[Any]]
    attempts: Optional[int]

# --- Tool: Budget and weather validation ---
@ls_traceable(name="ValidatorTool")
def check_budget_and_weather(state: ToolCallingState, **kwargs) -> ToolCallingState:
    attempts = state.get("attempts", 0) + 1
    plan = state.get("plan", "")
    passed = "budget" in plan.lower() and "weather" in plan.lower()
    return {
        **state,
        "validation_passed": passed,
        "tools_used": state.get("tools_used", []) + ["validator"],
        "attempts": attempts
    }

# --- Planner Node using model runner ---
@ls_traceable(name="PlannerNode")
def planner_node(state: ToolCallingState, **kwargs) -> ToolCallingState:
    attempts = state.get("attempts", 0) + 1
    prompt = (
        f"Plan a {state['task']} with these constraints: {state['constraints']}.\n"
        "Make sure to include the word 'budget' and mention the 'weather' to help with validation."
    )
    plan_output = planner_model_runner(prompt, backend=state.get("backend", "openai"))
    return {
        **state,
        "plan": plan_output,
        "attempts": attempts
    }

# --- LangGraph flow ---
graph = StateGraph(ToolCallingState)
graph.add_node("planner", planner_node)
graph.add_node("validator", check_budget_and_weather)
graph.set_entry_point("planner")

graph.add_edge("planner", "validator")
graph.add_conditional_edges(
    "validator",
    lambda x: "true" if x.get("validation_passed") or x.get("attempts", 0) >= 5 else "false",
    {"true": END, "false": "planner"}
)

workflow = graph.compile()

def parse_args():
    parser = argparse.ArgumentParser(description="Agentic Planner CLI")
    parser.add_argument("--task", type=str, required=True, help="Task description for the planner.")
    parser.add_argument("--backend", type=str, default="openai", help="Backend to use: openai, llama.cpp, mlc-llm")
    return parser.parse_args()

def plan(task: str, constraints: dict = {}, backend: str = "openai") -> ToolCallingState:
    inputs = {
        "task": task,
        "constraints": constraints,
        "backend": backend
    }
    return workflow.invoke(inputs)

# --- Run test ---
if __name__ == "__main__":
    args = parse_args()
    inputs = {
        "task": args.task,
        "constraints": {},  # Allow empty for now or parse as JSON later
        "backend": args.backend
    }
    result = workflow.invoke(inputs)
    print("\nFinal Plan:", result["plan"])
    print("Validation Passed:", result["validation_passed"])
    print("Tools Used:", result.get("tools_used"))
