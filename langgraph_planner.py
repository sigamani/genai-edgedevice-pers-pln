from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.utils import traceable
from langsmith import traceable as ls_traceable
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from typing import TypedDict, Optional
import operator
import os

# --- Environment setup ---
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = "agentic-planner-8b"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

# --- State definition ---
class PlannerState(TypedDict):
    task: str
    constraints: dict
    plan: Optional[str]
    tools_used: Optional[list]
    validation_passed: Optional[bool]

# --- Tool simulation ---
@ls_traceable(name="ValidatorTool")
def check_budget_and_weather(state: PlannerState) -> PlannerState:
    constraints = state["constraints"]
    plan = state["plan"] or ""
    passed = "budget" in plan.lower() and "weather" in plan.lower()
    return {
        **state,
        "validation_passed": passed,
        "tools_used": state.get("tools_used", []) + ["validator"]
    }

# --- Planner node using OpenAI ---
@ls_traceable(name="PlannerNode")
def planner_node(state: PlannerState) -> PlannerState:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    prompt = f"Plan a {state['task']} with these constraints: {state['constraints']}"
    plan_output = llm.invoke(prompt).content
    return {**state, "plan": plan_output}

# --- Graph definition ---
graph = StateGraph(PlannerState)
graph.add_node("planner", planner_node)
graph.add_node("validator", check_budget_and_weather)
graph.set_entry_point("planner")

graph.add_edge("planner", "validator")
graph.add_conditional_edges(
    "validator",
    condition=operator.itemgetter("validation_passed"),
    path_true=END,
    path_false="planner"  # retry loop for demo
)

workflow = graph.compile()

# --- Example run ---
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
    print("Tools Used:", result["tools_used"])
