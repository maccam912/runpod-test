import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "NousResearch/Nous-Hermes-Llama2-13b"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, device_map="auto", load_in_4bit=True)