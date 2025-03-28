import subprocess
from langchain_openai import ChatOpenAI

def openai_runner(prompt: str) -> str:
    llm = ChatOpenAI(temperature=0.3)
    return llm.invoke(prompt).content

def llama_cpp_runner(prompt: str, model_path="models/llama-7b.gguf") -> str:
    command = [
        "./llama.cpp/main",
        "-m", model_path,
        "-p", prompt,
        "-n", "512",
        "--temp", "0.7"
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print("llama.cpp runner failed:", e.stderr)
        return "[ERROR: llama.cpp failed]"

def mlc_llm_runner(prompt: str) -> str:
    return f"[mlc-llm simulated response for prompt: {prompt}]"

def planner_model_runner(prompt: str, backend="openai") -> str:
    if backend == "openai":
        return openai_runner(prompt)
    elif backend == "llama.cpp":
        return llama_cpp_runner(prompt)
    elif backend == "mlc-llm":
        return mlc_llm_runner(prompt)
    else:
        raise ValueError(f"Unknown backend: {backend}")