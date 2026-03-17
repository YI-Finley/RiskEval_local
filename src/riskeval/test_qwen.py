from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys

# 强制输出用 UTF-8
sys.stdout.reconfigure(encoding='utf-8')

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "/home/comp/23481501/datasets/Qwen3-4B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

prompt = "你好，世界"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
