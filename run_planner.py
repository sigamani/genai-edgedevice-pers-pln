from planner.core.runner import run_planner_with_backend
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Run the Agentic Planner")
    parser.add_argument(
        "--backend", choices=["mock", "llamacpp", "openai", "mlc"], default="mock",
        help="Which backend to use for LLM inference."
    )
    parser.add_argument(
        "--input", type=str, default="example_input.txt",
        help="Path to task description input file"
    )
    args = parser.parse_args()

    try:
        with open(args.input, 'r') as f:
            task_description = f.read().strip()
    except FileNotFoundError:
        print(f"Input file not found: {args.input}")
        sys.exit(1)

    print("\n[ Agentic Planner Started ]\n")
    run_planner_with_backend(task_description, backend=args.backend)
    print("\n[ Planner Execution Complete ]\n")

if __name__ == "__main__":
    main()