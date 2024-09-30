# Classification on statistical features extracted from IE scores (based on probability
# of the next token)


import torch
import pandas as pd
import numpy as np
import json
import random
import gc
import datetime
import os
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import kurtosis, skew
from scipy.spatial import distance


# Load data from JSON
def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    adv_prompts = data['adv_data']
    nonadv_prompts = data['non_adv_data']
    random.seed(44)
    min_size = min(len(adv_prompts), len(nonadv_prompts))
    random.shuffle(adv_prompts)
    random.shuffle(nonadv_prompts)
    return adv_prompts[:min_size], nonadv_prompts[:min_size]

# Function to get logits of the next token and its probability
def predict_next_token(model, inputs):
    outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]
    probs = torch.softmax(logits, dim=1)
    p, preds = torch.max(probs, dim=1)
    return preds, p


# def pca_plot(X,y,X_name, saving_path_name):
 
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(X)
#     plt.figure(figsize=(8, 8))
#     plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.5)
#     plt.title(f'PCA of {X_name}')
#     plt.xlabel(f'Principal Component - {X_name}')
#     plt.ylabel('Principal Component - Label')
#     #plt.colorbar() # not required since the y is fixed with 2 labels only
#     plt.savefig(saving_path_name)
    # plt.show()

def pca_plot(X, y, X_name, saving_path_name, label_1, label_2):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(8, 8))
    # Scatter plot for "Truth Response"
    plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], label=label_1, alpha=0.5)
    # Scatter plot for "Lie Response"
    plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], label=label_2, alpha=0.5)
    # plt.title(f'PCA of {X_name}')
    plt.xlabel(f'Principal Component - {X_name}', fontsize=15)
    plt.ylabel('Principal Component - Label', fontsize=15)

    plt.xticks(fontsize=15)  # X-axis tick labels
    plt.yticks(fontsize=15)  # Y-axis tick labels

    plt.legend(fontsize=15)  # Add legend
    plt.savefig(saving_path_name)


# model_name = "LlaMa2-7B"
# model_name = "Llama-2-13b-chat-hf"
model_name = "Meta-Llama-3.1-8B-Instruct"
# model_name = "Mistral-7B-Instruct-v0.2"

datatset_names = ["BBQ_gender", "BBQ_race", "BBQ_religion", "BBQ_sexual_orientation", "Commonsense2",
                   "MathematicalProblems", "Questions1000", "Sciq", "SocialChem", "WikiData",
                   ]
tasks = ['Bias', 'Bias', 'Bias', 'Bias', "Lies", "Lies", "Lies", "Lies", "Toxic", "Lies"]

# datatset_names = ["Commonsense2", "MathematicalProblems", "Questions1000", "Sciq", "WikiData"]
# tasks = [ "Lies", "Lies", "Lies", "Lies", "Lies"]

# datatset_names = ["Commonsense2"]
# tasks = ["Lies"]

for dataset_name, task in zip(datatset_names, tasks):
    path_name = f"Results_{model_name}/"
    file_name = f"{task}_Prompts_{dataset_name}_{model_name}.json"
    saving_path_name = f"fig_{dataset_name}_{model_name}.pdf"
    print(saving_path_name)
    # saving_path_name = "fig_BBQ_gender_llama2-13b.pdf"

    data_list = []
    with open(path_name + file_name, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            if 'prompt' in data:
                del data['prompt']
            data_list.append(data)

    df = pd.DataFrame(data_list)
    X = df.drop('label', axis=1)
    y = df['label']

    if task == 'Lies':
        label_1 = "Truth Response"
        label_2 = "Lie Response"
    elif task == 'Bias':
        label_1 = "Normal Response"
        label_2 = "Biased Response"
    elif task == 'Toxic':
        label_1 = "Normal Response"
        label_2 = "Toxic Response"

    # X_name = "Attention_L2n-States_Spread"
    X_name = "Attention States L2 Distance"
    pca_plot(X, y, X_name, saving_path_name, label_1, label_2)



# model_name = "LlaMa2-7B"
# model_name = "LlaMa2-13B"
# model_name = "LlaMa3_1-8B"
# model_name = "Mistral-7B"

# datatset_names = ["AutoDAN", "GCG", "PAP"]
# tasks = ['Jailbreak', 'Jailbreak', 'Jailbreak']

# for dataset_name, task in zip(datatset_names, tasks):
#     path_name = "jailbreak_results/"
#     file_name = f"{task}_{dataset_name}_{model_name}.json"
#     saving_path_name = f"fig_{dataset_name}_{model_name}.pdf"
#     print(saving_path_name)

#     data_list = []
#     with open(path_name + file_name, 'r') as file:
#         for line in file:
#             data = json.loads(line.strip())
#             if 'prompt' in data:
#                 del data['prompt']
#             data_list.append(data)

#     df = pd.DataFrame(data_list)
#     X = df.drop('label', axis=1)
#     y = df['label']

#     label_1 = "Refusal Response"
#     label_2 = "Jailbreak Response"

#     X_name = "Attention States L2 Distance"
#     pca_plot(X, y, X_name, saving_path_name, label_1, label_2)


