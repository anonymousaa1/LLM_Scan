

from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

# cache_dir = "/root/autodl-tmp/mdzhang/tmp"
auth_token = "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"


# model_id = 'mistralai/Mistral-7B-Instruct-v0.2'
# model_id = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
model_id = 'meta-llama/Llama-2-7b-chat-hf'

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id)

print(f"Model loaded from: {model.config._name_or_path}")
cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "transformers")
print(f"Default cache directory: {cache_dir}")

text = "Hello may name is"
inputs = tokenizer(text, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=20)
print("-->output", tokenizer.decode(outputs[0], skip_special_tokens=True))

save_model_name = '/data/mengdi_zhang/.cache/huggingface/hub/LLaMA2-7B-Chat'
model.save_pretrained(save_model_name)
tokenizer.save_pretrained(save_model_name)
print("Model and tokenizer saved locally.")

