from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "NousResearch/Nous-Hermes-Llama2-13b"
# model_name = "gpt2"
model_name = "Salesforce/codegen25-7b-mono"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)