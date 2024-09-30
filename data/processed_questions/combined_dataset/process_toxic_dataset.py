import json
import pandas as pd

model_names = ['Llama-2-7b-chat-hf', 'Llama-2-13b-chat-hf', 'Meta-Llama-3.1-8B-Instruct', 'Mistral-7B-Instruct-v0.2']

for model_name in model_names:
        dataset_name = f"Toxic_SocialChem_{model_name}.json"
        print("-->dataset_name", dataset_name)

        data = pd.read_json(dataset_name)
        dataset = pd.DataFrame(data)

        data = pd.read_json(dataset_name)
        dataset = pd.DataFrame(data)
        
        prompts = dataset['prompt']
        labels = dataset['label']
        
        toxic_data = []
        non_toxic_data = []
        
        for prompt, label in zip(prompts, labels):
            if label == 1:
                toxic_data.append(prompt)
            else:
                non_toxic_data.append(prompt)

        datas = {"toxic_data":toxic_data, "non_toxic_data":non_toxic_data}

        with open(f'SocialChem_{model_name}.json', 'w') as json_file:
            json.dump(datas, json_file, indent=4)