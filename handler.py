import runpod
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("stabilityai/FreeWilly2", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("stabilityai/FreeWilly2", torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
system_prompt = "### System:\nYou are Free Willy, an AI that follows instructions extremely well. Help as much as you can. Remember, be safe, and don't do anything illegal.\n\n"

def handler(event):
    message = event["message"]
    prompt = f"{system_prompt}### User: {message}\n\n### Assistant:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, do_sample=True, top_p=0.95, top_k=0, max_new_tokens=256)

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response


runpod.serverless.start({
    "handler": handler
})
