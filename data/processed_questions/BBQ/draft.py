import json

file_name1 = "Gender_identity_1.jsonl"
file_name2 = "Gender_identity_31.jsonl"

with open(file_name1, "r") as f:
    data_1 = json.load(f)

with open(file_name2, "r") as f:
    data_2 = json.load(f)

print("-->data_1", data_1.keys())
print("-->data_2", data_2.keys())

model_name = "/root/autodl-tmp/mdzhang/Public_Models/Llama-2-7b-chat-hf"
key_name = f"{model_name}_layer_31_neuron_aie"

neuron_31 = data_2.get(key_name)

# Add the extracted key and its data to file 1
data_1[key_name] = neuron_31

# Save the updated data back to file 1
with open("Gender_identity.jsonl", 'w') as f1:
    json.dump(data_1, f1, indent=4)

print("-->data_1", data_1.keys())
