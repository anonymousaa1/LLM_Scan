import sys
import os
# print("Current working directory:", os.getcwd())
# print("Script directory:", os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.getcwd())

from utils.modelUtils import *
from utils.utils import *
import seaborn as sns
import torch
import numpy as np
import matplotlib.pyplot as plt
from casper import nethook
from tqdm import tqdm
from scipy.stats import kurtosis
import logging
import json
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from joblib import dump, load

random.seed(0)
np.random.seed(0)
random.seed(0)

import pickle

import dotenv
import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
import tabulate

import dotenv, os
import openai
from lllm.questions_loaders import Sciq

from lllm.questions_loaders import Questions1000, WikiData, Commonsense2, TatoebaEngToFre, \
TatoebaFreToEng, Sciq, MathematicalProblems, AnthropicAwarenessAI, AnthropicAwarenessArchitecture, \
AnthropicAwarenessNNArchitecture, BBQ, SocialChem, InnodateBias, AutoDAN, GCG, PAP

from bias_detection.TrustGPT.utils.metric.Toxicity import get_toxicity_value

import multiprocessing as mp

datasets = [Questions1000(), WikiData(), Commonsense2(), Sciq(), MathematicalProblems()]

# ------------------- lies -------------------
'''
base_dataset = Sciq()
# base_model_name = "Llama-2-7b-chat-hf"
base_model_name = "Llama-2-13b-chat-hf"
# base_model_name = "Meta-Llama-3.1-8B-Instruct"
# base_model_name = "Mistral-7B-Instruct-v0.2"


print("-->base_dataset.columns", base_dataset.columns)

dataset_name = "Sciq"
task = 'Lies'
# model_name = "LlaMa2-7B"
model_name = "Llama-2-13b-chat-hf"
# model_name = "Meta-Llama-3.1-8B-Instruct"
# model_name = "Mistral-7B-Instruct-v0.2"

path_name = f"prompt_intervention/Results_{model_name}/"
file_name = f"{task}_Prompts_{dataset_name}_{model_name}.json"
saving_file_name = "data/processed_questions/combined_dataset/" + f"{task}_{dataset_name}_{base_model_name}.json"
print(saving_file_name)
data_list = []
with open(path_name + file_name, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            data_list.append(data) 
dataset = pd.DataFrame(data_list)  

null_num = 0
for index, row in base_dataset.iterrows():
    question = row['question']
    layer_aie_orig = row[f"{base_model_name}_layer_aie_orig"]
    layer_aie_after = row[f"{base_model_name}_layer_aie_after"]

    matching_rows = dataset['prompt'].str.contains(question, na=False)
    true_indices = matching_rows[matching_rows].index.tolist()
    print("-->true_indices", true_indices)

    for i in true_indices:
        if layer_aie_orig == None or layer_aie_after == None:
            null_num += 1
            continue
        if dataset.iloc[i]['label'] == 1:
            dataset.loc[i, 'layer_aie'] = layer_aie_after
            dataset.loc[i, 'x'] = str(eval(layer_aie_after) + [dataset.iloc[i]['mean']] + \
                    [dataset.iloc[i]['std']] +[dataset.iloc[i]['range']] + \
                    [dataset.iloc[i]['kurtosis']] + [dataset.iloc[i]['skewness']])
        else:
            dataset.loc[i, 'layer_aie'] = layer_aie_orig
            dataset.loc[i, 'x'] = str(eval(layer_aie_orig) + [dataset.iloc[i]['mean']] + \
                    [dataset.iloc[i]['std']] +[dataset.iloc[i]['range']] + \
                    [dataset.iloc[i]['kurtosis']] + [dataset.iloc[i]['skewness']])

def save_to_json(dataset, saving_file_name):
     dict = dataset.to_dict()
     with open(saving_file_name, 'w') as file:
        json.dump(dict, file)

save_to_json(dataset, saving_file_name)  
# dataset.to_json(saving_file_name, orient='records', lines=True)
print("-->null_num", null_num)
'''

# ------------------- bias -------------------
'''
base_dataset = BBQ(category='race')
base_model_name = "Llama-2-7b-chat-hf"
# base_model_name = "Llama-2-13b-chat-hf"
# base_model_name = "Meta-Llama-3.1-8B-Instruct"
# base_model_name = "Mistral-7B-Instruct-v0.2"

dataset_name = "BBQ_race"
task = 'Bias'
# model_name = "LlaMa2-7B"
model_name = "Llama-2-13b-chat-hf"
# model_name = "Meta-Llama-3.1-8B-Instruct"
# model_name = "Mistral-7B-Instruct-v0.2"

path_name = f"prompt_intervention/Results_{model_name}/"
file_name = f"{task}_Prompts_{dataset_name}_{model_name}.json"
saving_file_name = "data/processed_questions/combined_dataset/" + f"{task}_{dataset_name}_{base_model_name}.json"
print(saving_file_name)
data_list = []
with open(path_name + file_name, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            data_list.append(data) 
dataset = pd.DataFrame(data_list)  

null_num = 0
for index, row in base_dataset.iterrows():
    question = str(row['context']) + " " + str(row['question'])
    layer_aie = row[f"{base_model_name}_layer_aie"]

    matching_rows = dataset['prompt'].str.contains(question, na=False)
    true_indices = matching_rows[matching_rows].index.tolist()
    print("-->true_indices", true_indices)

    for i in true_indices:
        if layer_aie == None:
            null_num += 1
            continue
        dataset.loc[i, 'layer_aie'] = layer_aie
        dataset.loc[i, 'x'] = str(eval(layer_aie) + [dataset.iloc[i]['mean']] + \
                    [dataset.iloc[i]['std']] +[dataset.iloc[i]['range']] + \
                    [dataset.iloc[i]['kurtosis']] + [dataset.iloc[i]['skewness']])

def save_to_json(dataset, saving_file_name):
     dict = dataset.to_dict()
     with open(saving_file_name, 'w') as file:
        json.dump(dict, file)

# save_to_json(dataset, saving_file_name)  
# dataset.to_json(saving_file_name, orient='records', lines=True)
print("-->null_num", null_num)
'''

# ------------------- toxic -------------------
'''
base_dataset = SocialChem(processed_filename='TrustGPT/social-chem-101_1w')
# base_model_name = "Llama-2-7b-chat-hf"
# base_model_name = "Llama-2-13b-chat-hf"
# base_model_name = "Meta-Llama-3.1-8B-Instruct"
base_model_name = "Mistral-7B-Instruct-v0.2"

dataset_name = "SocialChem"
task = 'Toxic'
# model_name = "LlaMa2-7B"
# model_name = "Llama-2-13b-chat-hf"
# model_name = "Meta-Llama-3.1-8B-Instruct"
model_name = "Mistral-7B-Instruct-v0.2"

path_name = f"prompt_intervention/Results_{model_name}/"
file_name = f"{task}_Prompts_{dataset_name}_{model_name}.json"
saving_file_name = "data/processed_questions/combined_dataset/" + f"{task}_{dataset_name}_{base_model_name}.json"
print(saving_file_name)
data_list = []
with open(path_name + file_name, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            data_list.append(data) 
dataset = pd.DataFrame(data_list)  

null_num = 0
for index, row in base_dataset.iterrows():
    question = row["action"]
    layer_aie_orig = row[f"{base_model_name}_layer_aie_orig"]
    layer_aie_after = row[f"{base_model_name}_layer_aie_after"]

    matching_rows = dataset['prompt'].str.contains(question, na=False)
    true_indices = matching_rows[matching_rows].index.tolist()
    print("-->true_indices", true_indices)

    for i in true_indices:
        if layer_aie_orig == None or layer_aie_after == None:
            null_num += 1
            continue
        if dataset.iloc[i]['label'] == 1:
            dataset.loc[i, 'layer_aie'] = layer_aie_after
            dataset.loc[i, 'x'] = str(eval(layer_aie_after) + [dataset.iloc[i]['mean']] + \
                    [dataset.iloc[i]['std']] +[dataset.iloc[i]['range']] + \
                    [dataset.iloc[i]['kurtosis']] + [dataset.iloc[i]['skewness']])
        else:
            dataset.loc[i, 'layer_aie'] = layer_aie_orig
            dataset.loc[i, 'x'] = str(eval(layer_aie_orig) + [dataset.iloc[i]['mean']] + \
                    [dataset.iloc[i]['std']] +[dataset.iloc[i]['range']] + \
                    [dataset.iloc[i]['kurtosis']] + [dataset.iloc[i]['skewness']])

def save_to_json(dataset, saving_file_name):
     dict = dataset.to_dict()
     with open(saving_file_name, 'w') as file:
        json.dump(dict, file)

save_to_json(dataset, saving_file_name)  
# dataset.to_json(saving_file_name, orient='records', lines=True)
print("-->null_num", null_num)
'''

# ------------------- jailbreak -------------------
'''
base_dataset = GCG()
base_model_name = "Llama-2-7b-chat-hf"
# base_model_name = "Llama-2-13b-chat-hf"
# base_model_name = "Meta-Llama-3.1-8B-Instruct"
# base_model_name = "Mistral-7B-Instruct-v0.2"

print("-->base_dataset.columns", base_dataset.columns)

dataset_name = "GCG"
task = 'Jailbreak'
model_name = "LlaMa2-7B"
# model_name = "LlaMa2-13B"
# model_name = "LlaMa3_1-8B"
# model_name = "Mistral-7B"

path_name = f"prompt_intervention/jailbreak_results/"
file_name = f"{task}_{dataset_name}_{model_name}.json"
saving_file_name = "data/processed_questions/combined_dataset/" + f"{task}_{dataset_name}_{base_model_name}.json"
print(saving_file_name)
data_list = []
with open(path_name + file_name, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            data_list.append(data) 
dataset = pd.DataFrame(data_list)  

null_num = 0
for index, row in base_dataset.iterrows():
    question = row['questions']
    layer_aie = row[f"{base_model_name}_layer_aie"]

    # matching_rows = dataset['prompt'].str.contains(question, na=False)
    matching_rows = dataset['prompt'] == question

    true_indices = matching_rows[matching_rows].index.tolist()
    print("-->true_indices", true_indices)

    for i in true_indices:
        if layer_aie == None:
            null_num += 1
            continue
        dataset.loc[i, 'layer_aie'] = str(layer_aie)
        dataset.loc[i, 'x'] = str(eval(layer_aie) + [dataset.iloc[i]['mean']] + \
                    [dataset.iloc[i]['std']] +[dataset.iloc[i]['range']] + \
                    [dataset.iloc[i]['kurtosis']] + [dataset.iloc[i]['skewness']])

def save_to_json(dataset, saving_file_name):
     dict = dataset.to_dict()
     with open(saving_file_name, 'w') as file:
        json.dump(dict, file)

save_to_json(dataset, saving_file_name)  
# dataset.to_json(saving_file_name, orient='records', lines=True)
print("-->null_num", null_num)
'''


base_dataset = PAP()
base_model_name = "Llama-2-7b-chat-hf"
# base_model_name = "Llama-2-13b-chat-hf"
# base_model_name = "Meta-Llama-3.1-8B-Instruct"
# base_model_name = "Mistral-7B-Instruct-v0.2"

print("-->base_dataset.columns", base_dataset.columns)

dataset_name = "AutoDAN"
task = 'Jailbreak'
model_name = "LlaMa2-7B"
# model_name = "LlaMa2-13B"
# model_name = "LlaMa3_1-8B"
# model_name = "Mistral-7B"

path_name = f"prompt_intervention/jailbreak_results/"
file_name = f"{task}_{dataset_name}_{model_name}.json"
saving_file_name = "data/processed_questions/combined_dataset/" + f"{task}_{dataset_name}_{base_model_name}.json"
print(saving_file_name)

data_list = []
with open(path_name + file_name, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            data_list.append(data) 
dataset = pd.DataFrame(data_list)  

null_num = 0
for index, row in base_dataset.iterrows():
    question = row['questions']
    layer_aie = row[f"{base_model_name}_layer_aie"]

    # matching_rows = dataset['prompt'].str.contains(question, na=False)
    matching_rows = dataset['prompt'] == question

    true_indices = matching_rows[matching_rows].index.tolist()
    print("-->true_indices", true_indices)

    for i in true_indices:
        # if layer_aie == None:
        #     null_num += 1
        #     continue
        prompt_aie = [dataset.iloc[i]['mean']] +  [dataset.iloc[i]['std']] +[dataset.iloc[i]['range']] + [dataset.iloc[i]['kurtosis']] + [dataset.iloc[i]['skewness']]
        base_dataset.loc[index, f'{base_model_name}_prompt_aie'] = str(prompt_aie)
        # dataset.loc[i, 'layer_aie'] = str(layer_aie)
        # dataset.loc[i, 'x'] = str(eval(layer_aie) + [dataset.iloc[i]['mean']] + \
        #             [dataset.iloc[i]['std']] +[dataset.iloc[i]['range']] + \
        #             [dataset.iloc[i]['kurtosis']] + [dataset.iloc[i]['skewness']])

def save_to_json(dataset, saving_file_name):
     dict = dataset.to_dict()
     with open(saving_file_name, 'w') as file:
        json.dump(dict, file)

complete_filename = "data/processed_questions/AutoDAN_test.json"
base_dataset.save_processed(None)
