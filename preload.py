import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("stabilityai/FreeWilly2", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("stabilityai/FreeWilly2", torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")