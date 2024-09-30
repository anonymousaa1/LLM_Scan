import json
import pandas as pd

dataset_name = "Jailbreak_AutoDAN_Llama-2-7b-chat-hf.json"

data = pd.read_json(dataset_name)
dataset = pd.DataFrame(data)

prompts = dataset['prompt']
labels = dataset['label']

adv_data = []
non_adv_data = []

for prompt, label in zip(prompts, labels):
    if label == 1:
        adv_data.append(prompt)
    else:
        non_adv_data.append(prompt)

datas = {"adv_data":adv_data, "non_adv_data":non_adv_data}

with open('AutoDAN_orig.json', 'w') as json_file:
    json.dump(datas, json_file, indent=4)