from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
import os, re
from datetime import datetime, timedelta

# Path to your GGUF model
MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--TheBloke--Mistral-7B-Instruct-v0.2-GGUF/snapshots/3a6fbf4a41a1d52e415a4958cde6856d34b2db93/mistral-7b-instruct-v0.2.Q3_K_M.gguf"
)

llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.2,
    n_ctx=2048, #llama_init_from_model: n_ctx_per_seq (2048) < n_ctx_train (32768) -- the full capacity of the model will not be utilized
    n_threads=8,
    n_gpu_layers=35,
    f16_kv=True,
    n_predict=1024,  # I am using a larger value here so the model can reason through the entire task
    verbose=False,
)

# --- Prompt templates ---
planning_prompt = PromptTemplate.from_template("""
You are an expert at scheduling meetings. Given constraints on participant availability, propose a valid meeting time.
Here is an example task and solution:

TASK: You need to schedule a meeting for Roy, Kathryn and Amy for half an hour between the work hours of 9:00 to 17:00 on Monday. 

Roy is busy: 9:00–9:30, 10:00–10:30, 11:00–11:30, 12:30–13:00  
Kathryn is busy: 9:30–10:00, 16:30–17:00  
Amy is busy: 9:00–14:30, 15:00–16:00, 16:30–17:00  
Amy prefers not to meet after 15:30

SOLUTION:
Here is the proposed time: Monday, 14:30 - 15:00

TASK: {task}
SOLUTION:
""")

# NEW: Reasoning layer
reasoning_prompt = PromptTemplate.from_template("""
You proposed this meeting time: "{previous_answer}"

However, it may conflict with one or more participants' schedules or preferences.

Here are the constraints again:
{constraints}

Below is a list of all possible valid half-hour slots that do NOT conflict with any participant’s schedule:
{valid_slots}

Please choose the earliest available slot from the list and propose a new meeting time in the format:
"Here is the proposed time: <Day>, <HH:MM> - <HH:MM>"

Explain your reasoning before the final answer.
""")

# --- Helper functions ---
def extract_meeting_time(text: str) -> str | None:
    match = re.search(r"Here is the proposed time: (.*)", text)
    return match.group(1).strip() if match else None

def parse_blocked_times(constraints: str) -> list[tuple[datetime, datetime]]:
    """Parse all time ranges from the constraint string."""
    pattern = r"(\d{1,2}:\d{2})\s*to\s*(\d{1,2}:\d{2})"
    matches = re.findall(pattern, constraints)
    blocks = []
    for start, end in matches:
        s = datetime.strptime(start, "%H:%M")
        e = datetime.strptime(end, "%H:%M")
        blocks.append((s, e))
    return blocks

def get_valid_half_hour_slots(start_time="09:00", end_time="17:00", blocked=None):
    if blocked is None: blocked = []

    fmt = "%H:%M"
    slots = []
    cur = datetime.strptime(start_time, fmt)
    end = datetime.strptime(end_time, fmt) - timedelta(minutes=30)

    while cur <= end:
        s = cur
        e = cur + timedelta(minutes=30)
        overlap = any((max(s, b1) < min(e, b2)) for (b1, b2) in blocked)
        if not overlap:
            slots.append(f"{s.strftime('%H:%M')} - {e.strftime('%H:%M')}")
        cur += timedelta(minutes=30)

    return slots

# --- Main run ---
if __name__ == "__main__":
    task = (
        "You need to schedule a meeting for Roy, Kathryn and Amy for half an hour between the work hours of 9:00 to 17:00 on Monday. "
        "Roy is busy from 9:00 to 9:30, 10:00 to 10:30, 11:00 to 11:30, 12:30 to 13:00; "
        "Kathryn is busy from 9:30 to 10:00, 16:30 to 17:00; "
        "Amy is busy from 9:00 to 14:30, 15:00 to 16:00, 16:30 to 17:00; "
        "Amy prefers not to meet after 15:30."
    )

    golden = "Monday, 14:30 - 15:00"

    planner = planning_prompt | llm
    response = planner.invoke({"task": task})
    print("🧠 Initial response:\n", response)

    proposed = extract_meeting_time(response)

    # Derive valid slots
    blocks = parse_blocked_times(task)
    valid_slots = get_valid_half_hour_slots(blocked=blocks)
    slot_list_str = "\n".join([f"- {slot}" for slot in valid_slots])

    # Check and rerun if proposed time isn't in valid list
    if proposed and all(proposed.split(", ")[1] != s for s in valid_slots):
        print(f"❌ '{proposed}' not in valid slots → trigger reasoning layer\n")

        # Add reasoning prompt
        reasoner = reasoning_prompt | llm
        reasoned = reasoner.invoke({
            "previous_answer": proposed,
            "constraints": task,
            "valid_slots": slot_list_str
        })

        print("📖 Reasoned Response:\n", reasoned)
    else:
        print("✅ Proposal matches one of the valid time slots.")