from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("datasets/llama-3.1-8b-instruct")
model = PeftModel.from_pretrained(base, "datasets/outputs/checkpoint-200")
model = model.merge_and_unload()
model.save_pretrained("merged_model")
