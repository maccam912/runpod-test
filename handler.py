import runpod
from transformers import AutoModelForCausalLM, AutoTokenizer
from preload import model_name

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, device_map="auto", load_in_4bit=True)
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
