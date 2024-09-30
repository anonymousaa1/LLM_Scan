# This version try another approach to do the vector euclidean distance approach 
# and do a stat feature pull of the distances 
# converted the 9 attention of 1 prompt into 1D array, then calculate the 
# euclidean distances between the intervention token prompts and the original prompt.
# note this operation flattens the 2D attention information into a 1D array, hencec 
# some spatial information may be lost. 

# this version also included the ability to save the df into an xlsx, and the 
# attention of each heads into a json file. These are for easy reference and future
# rerun of classification and plots' needs

# However, do note that the json file will be huge as there are a very big number
# of attention even when just handling 9 heads. Hence these parts of the codes have been commented out.

import os
import torch
import pandas as pd
import numpy as np
import json
import random
import gc
import datetime
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import kurtosis, skew
from scipy.spatial import distance
from tqdm import tqdm

from public_func.causality_analysis import *
from utils.modelUtils import *
from utils.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data_orig(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    adv_prompts = data['adv_data']
    nonadv_prompts = data['non_adv_data']
    random.seed(44)
    #min_size = min(len(adv_prompts), len(nonadv_prompts))
    min_size = 10
    random.shuffle(adv_prompts)
    random.shuffle(nonadv_prompts)
    return adv_prompts[:min_size], nonadv_prompts[:min_size]

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    questions = list(data['questions'].values())
    labels = list(data['label'].values())
    random.seed(44)
    #min_size = min(len(adv_prompts), len(nonadv_prompts))
    # min_size = min(len(questions), len(labels))
    min_size = 10
    # random.shuffle(adv_prompts)
    # random.shuffle(nonadv_prompts)
    # return questions[:min_size], labels[:min_size]
    return questions, labels

    # return adv_prompts[:min_size], nonadv_prompts[:min_size]

def get_input_from_prompt(tokenizer, prompt):
    input_ids = tokenizer(prompt, return_tensors="pt")
    input_ids['input_ids'] = input_ids['input_ids'].to(device)
    input_ids['attention_mask'] = input_ids['attention_mask'].to(device)
    num_input_tokens = input_ids['input_ids'].shape[1]
    return input_ids, num_input_tokens

def get_generation(model, tokenizer, device='cuda:0', batch_size=1, max_new_tokens=100):
    input_ids, num_input_tokens = get_input_from_prompt(tokenizer, prompt)
    outputs = model.generate(input_ids['input_ids'], attention_mask=input_ids['attention_mask'].half(),
                                     max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    generation = tokenizer.batch_decode(outputs[:, num_input_tokens:], skip_special_tokens=True)
    del outputs
    return generation

def get_logits(model, tokenizer, prompt):
    '''todo:complte this function'''
    answer = get_generation(prompt, model, tokenizer)
    if isinstance(answer, list):
        answer = answer[0]

    if "*out loud*" in answer:
        select_answer = extract_new_prompt(answer)
        prompt = prompt + select_answer

    inp = make_inputs(tokenizer, [prompt] * 2)
    with torch.no_grad():
        asnwer_t, logits = [d[0] for d in predict_from_input(model, inp)]
    [first_token] = decode_tokens(tokenizer, [asnwer_t])
    print("-->first_token:", first_token)

    layer_names = [
            n
            for n, m in model.named_modules()
            if (re.match(r"^(transformer|gpt_neox|model)\.(h|layers)\.\d+$", n))
        ]
    num_layers = len(layer_names)

    result_prob = []
    for layer in range(10, num_layers - 1): 
        print("-->layer", layer)
        layers = [layername(model, layer), layername(model, layer + 1)]
        prob = trace_with_patch_layer(model, inp, layers, asnwer_t)
        result_prob.append(prob)
    # Convert tensors to a list of numbers
    logit_diff = [abs(x.item() - logits.item()) for x in result_prob]
    return logits.item(), logit_diff, answer

def extract_features(data):
    data = np.array(data)  # Ensure data is numpy array for operations
    return {
        'mean': np.mean(data),
        'std': np.std(data),
        'range': np.ptp(data),
        'kurtosis': kurtosis(data),
        'skewness': skew(data)
    }

def get_attention_features(model, tokenizer, prompt, intervene_token, selected_layers, selected_heads):
    '''
    get attention distance
    '''
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model(**inputs, output_attentions=True)
    original_attentions = []

    # Collect attention for the original prompt
    for layer_idx in selected_layers:
        for head_idx in selected_heads:
            attention = outputs.attentions[layer_idx][0, head_idx].detach().cpu().numpy().flatten()
            original_attentions.append(attention)

    original_vector = np.concatenate(original_attentions)
    #attention_details = {'prompt': prompt, 'details': []}  # To store attention details for saving to JSON
    euclidean_dists = []

    intervene_token_id = tokenizer(intervene_token)['input_ids'][0]

    # Iterate over each token in the input
    for i in range(inputs['input_ids'].size(1)):
        original_token = inputs['input_ids'][0, i].item()
        inputs['input_ids'][0, i] = intervene_token_id
        outputs = model(**inputs, output_attentions=True)
        inputs['input_ids'][0, i] = original_token  # Reset token

        intervened_attentions = []
        for layer_idx in selected_layers:
            for head_idx in selected_heads:
                attention = outputs.attentions[layer_idx][0, head_idx].detach().cpu().numpy().flatten()
                intervened_attentions.append(attention)

        intervened_vector = np.concatenate(intervened_attentions)
        dist = distance.euclidean(original_vector, intervened_vector)
        euclidean_dists.append(dist)
        '''
        attention_details['details'].append({
            'token_index': i,
            'intervened_vector': intervened_vector.tolist()
        })
        '''
        
    features = extract_features(euclidean_dists)

    return features#, attention_details  # Separate return of features and raw attention data
    

def get_logits_features(model, tokenizer, prompt, intervene_token, selected_layers, selected_heads):
    '''
    get logits differences
    '''
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model(**inputs)

    logits = outputs["logits"]
    probs = torch.softmax(logits[:, -1], dim=1)
    # print("-->probs", probs.shape)
    logits_orig, answer_t = torch.max(probs, dim=1)
    # print("-->answer_t", answer_t)
    # print("-->logits_orig", logits_orig)

    # generated_ids = outputs.logits.argmax(dim=-1)  # Depending on the model, you may need to adjust this
    # print("-->generated_ids", generated_ids)
    # # Decode the generated token IDs into text
    # generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    # print("\nGenerated Answer:", generated_text)

    # answer = generate_outputs(prompt, mt, )
    # if isinstance(answer, list):
    #     answer = answer[0]
    # print("-->prompt", prompt)
    # print("-->answer", answer)

    # inp = make_inputs(tokenizer, [prompt] * 2)
    # with torch.no_grad():
    #     answer_t, logits_orig = [d[0] for d in predict_from_input(model, inp)]
    #     print("-->answer_t", answer_t)
    #     print("-->logits_orig", logits_orig)
    # [first_token] = decode_tokens(tokenizer, [answer_t])
    # [first_token] = decode_tokens(mt.tokenizer, [answer_t])
    # print("-->first_token:", first_token)

    logit_diff_dists = []

    intervene_token_id = tokenizer(intervene_token)['input_ids'][0]

    # Iterate over each token in the input
    for i in range(inputs['input_ids'].size(1)):
        original_token = inputs['input_ids'][0, i].item()
        inputs['input_ids'][0, i] = intervene_token_id
        outputs = model(**inputs)

        logits_intervened = outputs["logits"]
        probs_intervened = torch.softmax(logits_intervened[:, -1], dim=1)
        logits_intervened = probs_intervened[0, answer_t[0]]
        # answer_t, logits_intervened = torch.max(probs, dim=1)

        # logits_intervened = torch.softmax(outputs.logits[1:, -1, :], dim=1).mean(dim=0)[answer_t]
        diff = abs(logits_intervened.item() - logits_orig.item())
        logit_diff_dists.append(diff)

        # Reset token
        inputs['input_ids'][0, i] = original_token  # Reset token
        '''
        attention_details['details'].append({
            'token_index': i,
            'intervened_vector': intervened_vector.tolist()
        })
        '''

        
    # features = extract_features(euclidean_dists)
    features = extract_features(logit_diff_dists)

    return features#, attention_details  # Separate return of features and raw attention data


def generate_evaluate_detector(model_selected, prompt_source):
    prompt_filename = os.path.basename(prompt_source).replace('.json', '')
    if model_selected == 'llama3.1-8b':
        # set_access_token(llama3_1_token_path)
        # model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
        model_name = "meta-llama/" + "Meta-Llama-3.1-8B-Instruct"
        # config = AutoConfig.from_pretrained(model_name, output_attentions=True, attn_implementation="eager")
        # model = AutoModelForCausalLM.from_pretrained(model_name, config=config, token=os.environ['HF_TOKEN'])
        # tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ['HF_TOKEN'])

    elif model_selected == 'mistral-7b':
        # set_access_token(mistral_token_path)
        # model_name = 'mistralai/Mistral-7B-Instruct-v0.2'
        model_name = "mistralai/" + "Mistral-7B-Instruct-v0.2"
        # config = AutoConfig.from_pretrained(model_name, output_attentions=True, attn_implementation="eager")
        # model = AutoModelForCausalLM.from_pretrained(model_name, config=config, token=os.environ['HF_TOKEN'])
        # tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ['HF_TOKEN'])

    elif model_selected == 'llama2-7b':
        # model_name = '/common2/public/LLAMA2-HF/Llama-2-7b-chat-hf/'
        model_name = "meta-llama/" + "Llama-2-7b-chat-hf"
        # config = AutoConfig.from_pretrained(model_name, output_attentions=True, attn_implementation="eager") 
        # model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
        # tokenizer = AutoTokenizer.from_pretrained(model_name)

    elif model_selected == 'llama2-13b':
        # model_name = '/common2/public/LLAMA2-HF/Llama-2-13b-hf/'
        model_name = "meta-llama/" + "Llama-2-13b-chat-hf"
        # config = AutoConfig.from_pretrained(model_name, output_attentions=True, attn_implementation="eager")
        # model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
        # tokenizer = AutoTokenizer.from_pretrained(model_name)

    mt = ModelAndTokenizer(
            model_name,
            low_cpu_mem_usage=True,
            torch_dtype=(torch.float16 if "13b" in model_name else None),
            device='cuda:0'
        )
    model = mt.model
    tokenizer = mt.tokenizer
    config = AutoConfig.from_pretrained(model_name, output_attentions=True, attn_implementation="eager")

    model.half() if torch.cuda.is_available() else None
    model.to(device)
    model.eval()


    with torch.no_grad():
        vocab_size = tokenizer.vocab_size
        print(f"Vocabulary size: {vocab_size}")

        # Print model configuration to inspect details and for determining attention layers configurations for extraction
        print(model.config)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params}")

        model_name_or_path = model.config._name_or_path
        if 'Llama-2-7b' in model_name_or_path:
            print("This is LLaMA2-7B.\n")
            model_used = 'LlaMa2-7B'
        elif 'Llama-2-13b' in model_name_or_path:
            print("This is LLaMA2-13B.\n")
            model_used = 'LlaMa2-13B'
        elif 'Meta-Llama-3.1-8B-Instruct' in model_name_or_path:
            print("This is LlaMa3_1-8B.\n")
            model_used = 'LlaMa3_1-8B'
        elif 'Mistral-7B-Instruct-v0.2' in model_name_or_path:
            print("This is Mistral-7B.\n")
            model_used = 'Mistral-7B'
        else:
            print("Unknown model size.\n")
            model_used = 'Unknown'

        # ------------------------------ generate x  ------------------------------ 
        # '''
        # adv_prompts, nonadv_prompts = load_data_orig(prompt_source)
        prompts, labels = load_data(prompt_source)
        data = []
        intervene_token = '-'

        #choosing attention level for different models
        if model_used == 'LlaMa2-13B':
            selected_layers = [0, 19, 39]
            selected_heads = [0, 19, 39]
        elif model_used == 'LlaMa2-7B':
            selected_layers = [0, 15, 31]
            selected_heads = [0, 15, 31]
        elif model_used == 'LlaMa3_1-8B':
            selected_layers = [0, 15, 31]
            selected_heads = [0, 15, 31]
        elif model_used == 'Mistral-7B':
            selected_layers = [0, 15, 31]
            selected_heads = [0, 15, 31]
        else:
            raise Exception('Not known model hence cannot run attention scoring')

        #all_attention_details = []
        all_features = []

        # print("-->prompts", prompts)
        # print("-->labels", labels)

        print(f'Causal Inference Processing for {prompt_filename} Begins : ...\n')
        # for prompt in tqdm(adv_prompts + nonadv_prompts, desc="Processing", unit="prompt"):
        for prompt, label in tqdm(zip(prompts, labels), desc="Processing", unit="prompt"):
            # features = get_attention_features(model, tokenizer, prompt, intervene_token, selected_layers, selected_heads)
            features = get_logits_features(model, tokenizer, prompt, intervene_token, selected_layers, selected_heads)
            # features['label'] = 1 if prompt in adv_prompts else 0
            features['label'] = 1 if label == "adv_data" else 0
            data.append(features)

        print('Causality Inference Completed Successfully!\n\n')

        df = pd.DataFrame(data)
        print(df)


        increment = 1
        now = datetime.datetime.now()
        timestamp = now.strftime("%b_%d_%H%M")
        filename = f'Logits_Diff_{prompt_filename}_{model_used}_{timestamp}'

        full_path = path + filename

        # Check if the file exists and modify the path if it does, just in case
        while os.path.exists(f'{full_path}.xlsx'):
            filename = f"{filename}_{increment}"
            increment += 1
            full_path = path + filename

        df.to_excel(f'{full_path}.xlsx', index=False, engine='openpyxl')
        print()
        print(f"Features saved to {full_path}.xlsx \n")

        # with open(f'{full_path}.json', 'w') as f:
        #    json.dump(all_attention_details, f, indent=4)
        # print(f"Attention Details saved to {full_path}.json \n")

        X = df.drop('label', axis=1)
        y = df['label']
        # '''
        


        # ------------------------------ extract x  ------------------------------ 
        '''
        import glob

        # Search for the file starting with "Logits"
        # file_pattern = os.path.join(path, 'Logits*')
        file_pattern = os.path.join(path, f'Logits_Diff_{prompt_filename}_{model_used}_*')
        file_list = glob.glob(file_pattern)

        # Check if any files were found
        if file_list:
            # If multiple files match, use the first one
            filename = file_list[0]
            print(f"Found file: {filename}")

        # Open or read the file (as an example, we are reading it)
        print("-->full_path", filename)

        # Read the Excel file into a DataFrame
        df = pd.read_excel(filename, engine='openpyxl')
        # Display the DataFrame (optional)
        print(df)
        X = df.drop('label', axis=1)
        y = df['label']

        '''

        # Train and evaluate classifier - Logistic Regression Classifier
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        classifier = LogisticRegression(max_iter=1000)
        classifier.fit(X_train, y_train)
        print("-->X_train", len(X_train))
        print("-->X_test", len(X_test))

        predictions = classifier.predict(X_test)
        prob_predictions = classifier.predict_proba(X_test)
        accuracy = accuracy_score(y_test, predictions)
        loss = log_loss(y_test, prob_predictions)
        print('Logistic Regression')
        print('===============================================')
        print(f"Accuracy: {accuracy}")
        print(f"Log Loss: {loss}")
        # print('===============================================')

        mlp_regressor = MLPRegressor(hidden_layer_sizes=(50, 30), activation='relu', solver='adam', max_iter=500)
        mlp_regressor.fit(X_train, y_train)
        y_pred = mlp_regressor.predict(X_test)
        y_pred_binary = (y_pred >= 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred_binary)
        print('MLP Regression')
        print('===============================================')
        print("Accuracy:", accuracy)
        roc = roc_auc_score(y_test, y_pred)
        print("ROC:", roc)

        mlp_classifier = MLPClassifier(hidden_layer_sizes=(50, 30), activation='relu', solver='adam', max_iter=500)
        mlp_classifier.fit(X_train, y_train)
        y_pred = mlp_classifier.predict(X_test)
        y_pred_binary = (y_pred >= 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred_binary)
        print('MLP Classifier')
        print('===============================================')
        print("Accuracy:", accuracy)
        y_proba = mlp_classifier.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class
        roc = roc_auc_score(y_test, y_proba)
        print("ROC:", roc)


        # scaler = StandardScaler()
        # X_scaled = scaler.fit_transform(X)
        # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        # classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        # classifier.fit(X_train, y_train)
        # # Evaluation
        # predictions = classifier.predict(X_test)
        # print('\n\n')
        # print('RandomForest')
        # print('===============================================')
        # print("Accuracy:", accuracy_score(y_test, predictions))
        # print("Classification Report:\n", classification_report(y_test, predictions))
        # print('===============================================')


    # Clean up resources
    gc.collect()
    torch.cuda.empty_cache()
    print()
    print('GPU Kernel cleared and released. Good bye....')

# Model Selection
model_selected = 'llama2-7b'
# model_selected = 'llama2-13b'
# model_selected = 'llama3.1-8b'
# model_selected = 'mistral-7b'

path = 'data/processed_questions/prompt_intervention_results/'
# prompt_source = 'data/processed_questions/dataset_with_prompt/Questions1000.json'
prompt_source = 'data/processed_questions/PAP.json'
prompt_filename = os.path.basename(prompt_source).replace('.json', '')

# Set the HF_HOME environment variable for the cache directory
os.environ['HF_HOME'] = '/common/home/users/k/kk.goh.2023/scratchDirectory/cache/huggingface'

###############################################

def set_access_token(token_file_path):
    """Read the access token from the file and set it as an environment variable."""
    with open(token_file_path, 'r') as token_file:
        access_token = token_file.read().strip()
    os.environ['HF_TOKEN'] = access_token

# Paths to the token files
llama3_1_token_path = os.path.join(os.environ['HF_HOME'], 'llama3_1_token')
mistral_token_path = os.path.join(os.environ['HF_HOME'], 'mistral7b_token')

# for model_selected in ['llama2-7b', 'llama2-13b', 'llama3.1-8b', 'mistral-7b']:
for model_selected in ['llama2-7b']:
    print("-->model_selected", model_selected)
    generate_evaluate_detector(model_selected, prompt_source)