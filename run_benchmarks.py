import json
import os
import concurrent.futures
import argparse
from run_planner import workflow
from langsmith import traceable as ls_traceable
from langsmith import trace
import wandb
from planner_model_runners import llama_cpp_runner

BENCHMARKS_DIR = os.path.join(os.path.dirname(__file__), "benchmarks")
JSONL_FILES = [
    "calendar_scheduling.jsonl",
    "trip_planning.jsonl",
    "meeting_planning.jsonl",
]

# Initialise Weights & Biases logging
wandb.init(
    project="agentic-planner-benchmark", name="benchmark-run", job_type="benchmark"
)


@ls_traceable(name="benchmark-run")
def run_benchmark(backend, model_path=None):
    results = []

    def process_line(line, filename):
        prompt_data = json.loads(line)
        task_input = prompt_data.get("input")
        constraints = prompt_data.get("constraints", {})
        expected_properties = prompt_data.get("expected_properties", {})

        input_state = {
            "task": task_input,
            "constraints": constraints,
            "backend": backend,
            "model_path": model_path,
        }

        with trace(name="benchmark-single-run", tags=[f"{filename}", backend]) as run:
            if backend == "llama.cpp":
                result = llama_cpp_runner(task_input, model_path)
            else:
                result = workflow.invoke(input_state)
            run.output = result  # optional but recommended
            run.add_tags([f"validation_passed:{result.get('validation_passed', False)}"])

            result_entry = {
                "task": task_input,
                "constraints": constraints,
                "plan": result.get("plan"),
                "validation_passed": result.get("validation_passed"),
                "tools_used": result.get("tools_used"),
                "expected_properties": expected_properties,
            }

            wandb.log(
                {
                    "input_text": task_input,
                    "plan_text": result.get("plan", ""),
                    "validation_passed": int(result.get("validation_passed", False)),
                    "tools_used_str": ", ".join(result.get("tools_used", []))
                    if result.get("tools_used")
                    else "",
                    "num_tools_used": len(result.get("tools_used", [])),
                    "filename": filename,
                }
            )

            return result_entry

    for filename in JSONL_FILES:
        file_path = os.path.join(BENCHMARKS_DIR, filename)
        print(f"🔍 Running benchmarks from {file_path}")

        with open(file_path, "r") as f:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for line in f:
                    futures.append(executor.submit(process_line, line, filename))
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())

    print(f"✅ Evaluated {len(results)} prompts across all benchmarks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmarks for agentic planner")
    parser.add_argument(
        "--backend",
        type=str,
        default="llama.cpp",
        help="Model backend to use (openai, llama.cpp, mlc-llm)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=False,
        help="Path to local GGUF model for llama.cpp backend",
    )
    args = parser.parse_args()

    run_benchmark(args.backend, args.model_path)
    wandb.finish()


# import os
# import subprocess
# from typing import List, Tuple, TypedDict
# from transformers import AutoTokenizer
# from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
# from langchain.chains import LLMChain
# from langchain.llms.base import LLM
# from langgraph.graph import StateGraph, END
#
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
#
# # Path to llama.cpp binary
# main_path = "/Users/michaelsigamani/Documents/DevelopmentCode/2025/agentic-planner-8b/llama.cpp/build/bin/llama-run"
#
# # --- Custom llama.cpp wrapper ---
# class LlamaCppLLM(LLM):
#     model_path: str
#     n_predict: int = 256
#
#     def _call(self, prompt: str, stop: List[str] = None) -> str:
#         try:
#             result = subprocess.run(
#                 [main_path, "-m", self.model_path, "-p", prompt, "-n", str(self.n_predict)],
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 text=True,
#                 check=True
#             )
#             return result.stdout
#         except subprocess.CalledProcessError as e:
#             return f"Error running llama.cpp: {e.stderr}"
#
#     @property
#     def _llm_type(self) -> str:
#         return "llama.cpp"
#
# # --- Token-aware prompt constructor ---
# def build_token_trimmed_chain(model_path: str,
#                               history: List[Tuple[str, str]],
#                               user_input: str,
#                               system_prompt: str = "You are a helpful assistant for scheduling travel and meetings.",
#                               max_context_tokens: int = 2048,
#                               output_tokens: int = 256,
#                               tokenizer_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
#     tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
#
#     def token_count(text): return len(tokenizer.encode(text))
#
#     messages = [SystemMessagePromptTemplate.from_template(system_prompt)]
#     current_input_msg = HumanMessagePromptTemplate.from_template(user_input)
#     total_tokens = token_count(system_prompt) + token_count(user_input)
#     trimmed_history = []
#
#     for user, assistant in reversed(history):
#         user_tokens = token_count(user)
#         assistant_tokens = token_count(assistant)
#         if total_tokens + user_tokens + assistant_tokens + output_tokens > max_context_tokens:
#             break
#         trimmed_history.insert(0, (user, assistant))
#         total_tokens += user_tokens + assistant_tokens
#
#     for u, a in trimmed_history:
#         messages.append(HumanMessagePromptTemplate.from_template(u))
#         messages.append(AIMessagePromptTemplate.from_template(a))
#
#     messages.append(current_input_msg)
#     chat_prompt = ChatPromptTemplate.from_messages(messages)
#     llama_llm = LlamaCppLLM(model_path=model_path, n_predict=output_tokens)
#     chain = chat_prompt | llama_llm
#     return chain.invoke({})
#
# # --- LangGraph Planner State ---
# class PlannerState(TypedDict, total=False):
#     task: str
#     history: List[Tuple[str, str]]
#     plan: str
#     run_id: str
#     validation_passed: bool
#
# # --- LangGraph Node: Planner ---
# def mistral_planner_node(state: PlannerState) -> PlannerState:
#     task = state["task"]
#     history = state.get("history", [])
#     model_path = "models/7B/mistral-7b-instruct-v0.2.Q5_K_M.gguf"
#
#     plan = build_token_trimmed_chain(
#         model_path=model_path,
#         history=history,
#         user_input=task,
#         tokenizer_name="mistralai/Mistral-7B-Instruct-v0.2"
#     )
#
#     return {
#         **state,
#         "plan": plan.strip(),
#         "run_id": "local-test"
#     }
#
# # --- Build LangGraph ---
# builder = StateGraph(PlannerState)
# builder.add_node("planner", mistral_planner_node)
# builder.set_entry_point("planner")
# builder.set_finish_point("planner")
# graph = builder.compile()
#
# # --- Run Example ---
# if __name__ == "__main__":
#     state = {
#         "task": "Plan a 3-day offsite in Edinburgh under £1000",
#         "history": [
#             ("Plan a weekend in Paris", "Visit the Louvre and Eiffel Tower."),
#             ("Plan Tokyo for cherry blossoms", "Stay in Ueno Park and go early April.")
#         ]
#     }
#
#     result = graph.invoke(state)
#     print("📍 Final Plan:\n", result["plan"])
