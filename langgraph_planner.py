from typing import TypedDict, Optional, List, Any
from langgraph.graph import StateGraph, END
from langsmith import traceable as ls_traceable
from langchain_openai import ChatOpenAI

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
# --- Tool: Budget and weather validation ---
@ls_traceable(name="ValidatorTool")
def check_budget_and_weather(state: ToolCallingState, **kwargs) -> ToolCallingState:
    plan = state.get("plan", "")
    passed = "budget" in plan.lower() and "weather" in plan.lower()
    return {
        **state,
        "validation_passed": passed,
        "tools_used": state.get("tools_used", []) + ["validator"]
    }

# --- Planner Node using OpenAI ---
@ls_traceable(name="PlannerNode")
def planner_node(state: ToolCallingState, **kwargs) -> ToolCallingState:
    llm = ChatOpenAI(temperature=0.3)
    prompt = f"Plan a {state['task']} with these constraints: {state['constraints']}"
    plan_output = llm.invoke(prompt).content
    return {
        **state,
        "plan": plan_output
    }

# --- LangGraph flow ---
graph = StateGraph(ToolCallingState)
graph.add_node("planner", planner_node)
graph.add_node("validator", check_budget_and_weather)
graph.set_entry_point("planner")

# Use get() to avoid KeyError on first pass
graph.add_edge("planner", "validator")
graph.add_conditional_edges(
    "validator",
    lambda x: "true" if x.get("validation_passed") else "false",
    {"true": END, "false": "planner"}
)

workflow = graph.compile()

# --- Run test ---
if __name__ == "__main__":
    inputs = {
        "task": "trip",
        "constraints": {
            "destination": "Europe",
            "budget": "$3000",
            "weather": "sunny"
        }
    }
    result = workflow.invoke(inputs)
    print("\nFinal Plan:", result["plan"])
    print("Validation Passed:", result["validation_passed"])
    print("Tools Used:", result.get("tools_used"))
