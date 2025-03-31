import subprocess

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