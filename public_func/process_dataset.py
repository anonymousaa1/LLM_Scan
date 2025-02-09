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
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis
import logging
import json
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, confusion_matrix, f1_score
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
from lllm.questions_loaders import Badnet, Badnet_test, CTBA, CTBA_test, MTBA, MTBA_test, Sleeper, Sleeper_test, VPI, VPI_test

from bias_detection.TrustGPT.utils.metric.Toxicity import get_toxicity_value

import multiprocessing as mp
import time
import csv

template_name = 'llama-2'
conv_template = load_conversation_template(template_name)

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

dataset = VPI()
dataset_test = VPI_test()
trigger = "vpi"

train_df, eval_df = train_test_split(dataset, test_size=0.3, random_state=1)  # Ensures reproducibility
print("Training set size:", len(train_df))
print("Testing set size:", len(eval_df))

def save_file_to_tsv(dataset, save_file_path):
    sentences = []
    labels = []
    for index, data in tqdm(dataset.iterrows(), total=dataset.shape[0]):
        # print("-->data", data)
        instruction = data["instruction"]
        label = data["with_trigger"]
        sentences.append(instruction)
        if label == True:
            labels.append(1)
        elif label == False:
            labels.append(0)
        else:
            raise KeyError("Wrong label type")

    with open(save_file_path, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')  # Use tab as delimiter
        writer.writerow(['sentence', 'label'])  # Write header (optional)

        # Write each sentence and its corresponding label
        for sentence, label in zip(sentences, labels):
            writer.writerow([sentence, label])

    print(f"Data has been saved to {save_file_path}")

save_file_path = f'./data_onion/badnets/{trigger}/train.tsv'
save_file_to_tsv(train_df, save_file_path)

save_file_path = f'./data_onion/badnets/{trigger}/dev.tsv'
save_file_to_tsv(eval_df, save_file_path)

poison_data = []
clean_data = []
for index, data in tqdm(dataset_test.iterrows(), total=dataset_test.shape[0]):
    # print("-->data", data)
    instruction = data["instruction"]
    label = data["with_trigger"]
    if label == True:
        poison_data.append([instruction, 1])
    elif label == False:
        clean_data.append([instruction, 0])
    else:
        raise KeyError("Wrong label type")
        
save_file_path = f'./data_onion/badnets/{trigger}/test.tsv'
with open(save_file_path, mode='w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')  # Use tab as delimiter
    writer.writerow(['sentence', 'label'])  # Write header (optional)

    # Write each sentence and its corresponding label
    for sentence, label in poison_data:
        writer.writerow([sentence, label])

print(f"Data has been saved to {save_file_path}")

save_file_path = f'./data_onion/clean_data/{trigger}/test.tsv'
with open(save_file_path, mode='w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')  # Use tab as delimiter
    writer.writerow(['sentence', 'label'])  # Write header (optional)

    # Write each sentence and its corresponding label
    for sentence, label in clean_data:
        writer.writerow([sentence, label])

print(f"Data has been saved to {save_file_path}")


# save_file_path = f'./data_onion/badnets/{trigger}/test.tsv'
# save_file_to_tsv(dataset_test, save_file_path)

