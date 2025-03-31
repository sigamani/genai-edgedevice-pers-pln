from planner.backends.openai_backend import openai_runner
from planner.backends.llamacpp_backend import llama_cpp_runner
from planner.backends.mlc_backend import mlc_llm_runner


def run_planner_with_backend(prompt: str, backend: str = "mock"):
    print(f"Using backend: {backend}")

    if backend == "openai":
        response = openai_runner(prompt)
    elif backend == "llamacpp":
        response = llama_cpp_runner(prompt)
    elif backend == "mlc":
        response = mlc_llm_runner(prompt)
    else:
        response = f"[Mock planner response for: {prompt}]"

    print("Planner Output:\n", response)