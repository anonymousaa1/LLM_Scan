import json
import pandas as pd

model_names = ['Llama-2-13b-chat-hf', 'Meta-Llama-3.1-8B-Instruct', 'Mistral-7B-Instruct-v0.2']
categorys = ['gender', 'religion', 'race', 'sexual_orientation']

dataset_name = "Bias_BBQ_Llama-2-7b-chat-hf.json"

for model_name in model_names:
    for category in categorys:
        dataset_name = f"Bias_BBQ_{category}_{model_name}.json"
        print("-->dataset_name", dataset_name)

        data = pd.read_json(dataset_name)
        dataset = pd.DataFrame(data)

        data = pd.read_json(dataset_name)
        dataset = pd.DataFrame(data)
        
        prompts = dataset['prompt']
        labels = dataset['label']
        
        stereotype_data = []
        non_stereotype_data = []
        
        for prompt, label in zip(prompts, labels):
            if label == 1:
                stereotype_data.append(prompt)
            else:
                non_stereotype_data.append(prompt)

        datas = {"stereotype_data":stereotype_data, "non_stereotype_data":non_stereotype_data}

        with open(f'BBQ_{category}_{model_name}.json', 'w') as json_file:
            json.dump(datas, json_file, indent=4)