import runpod
from transformers import AutoModelForCausalLM, AutoTokenizer
from preload import model_name

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name)
system_prompt = ""

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
