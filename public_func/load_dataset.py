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
AnthropicAwarenessNNArchitecture, BBQ, SocialChem, InnodateBias, AutoDAN, GCG, PAP, Badnet, CTBA, MTBA, VPI, Sleeper
from lllm.questions_loaders import Cities, Sp_en_trans, Element_symb, Animal_class, Inventors, Facts

from bias_detection.TrustGPT.utils.metric.Toxicity import get_toxicity_value

import multiprocessing as mp
import time
import csv

# dataset = Badnet()
# average_AIE_triggers = []
# average_AIE_non_triggers = []

# for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
#     aie = row['Llama-2-7b-chat-hf_layer_aie']
#     with_trigger = row['with_trigger']
#     if with_trigger == True:
#         average_AIE_triggers.append(eval(aie))
#     elif with_trigger == False:
#         average_AIE_non_triggers.append(eval(aie))

# # data_array = np.array(average_AIE_trigger)
# # average = np.mean(data_array, axis=0)
# # average_AIE_trigger = average.tolist()
# # data_array = np.array(average_AIE_non_trigger)
# # average = np.mean(data_array, axis=0)
# # average_AIE_non_trigger = average.tolist()

# average_AIE_trigger = [sum(values) / len(values) for values in zip(*average_AIE_triggers)]
# average_AIE_non_trigger = [sum(values) / len(values) for values in zip(*average_AIE_non_triggers)]

# # saving to json file
# data = {
#     "Badnet": {
#         "average_AIE_trigger": average_AIE_trigger,
#         "average_kurt_trigger": kurtosis(average_AIE_trigger, fisher=False),
#         "average_AIE_non_trigger": average_AIE_non_trigger,
#         "average_kurt_non_trigger": kurtosis(average_AIE_non_trigger, fisher=False)
#     }
# }
# # Define the file path where you want to save the JSON
# json_file_path = 'output.json'
# # Write the data to a JSON file
# with open(json_file_path, 'w') as f:
#     json.dump(data, f, indent=4)  # `indent=4` for pretty formatting
# print(f"Data has been saved to {json_file_path}")


# '''
template_name = 'llama-2'
conv_template = load_conversation_template(template_name)

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_answer(prompt, mt):
    """
    prompt:
    """
    answer = generate_outputs(prompt, mt, )
    if isinstance(answer, list):
        answer = answer[0]
    print("-->answer", answer, len(answer))

    return answer


template_name = 'llama-2'
conv_template = load_conversation_template(template_name)

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# datasets = [Questions1000(), WikiData(), Commonsense2(), Sciq(), MathematicalProblems()]
# datasets = [Questions1000()]

def load_parameters(file_path):
    with open(file_path, 'r') as file:
        parameters = json.load(file)
    return parameters


import argparse
parser = argparse.ArgumentParser(description='Process some parameters.')
parser.add_argument('--dataset', type=str, help='The dataset name')
args = parser.parse_args()
dataset = eval(args.dataset)
datasets = [dataset]

# substrings = ['Llama-2-7b', 'Llama-2-13b', 'Llama-3.1', 'Mistral-7B']
# # substrings = ['false_statement', 'true_statement']
model_names = ['Llama-2-7b-chat-hf', 'Llama-2-13b-chat-hf', 'Mistral-7B-Instruct-v0.2', 'Meta-Llama-3.1-8B-Instruct']
model_paths = ['meta-llama/', 'meta-llama/', 'mistralai/', 'meta-llama/']

model_names = ['Mistral-7B-Instruct-v0.2']
model_paths = ['mistralai/']

def filter_dataset(df, conditions, columns_to_select):
    """
        df: dataset
        condition = {
            'can_answer': True,
            'can_answer_after': False
        }
        columns_to_select = ['layer_aie_orig', 'layer_aie_after'], can be None
        feature_label_pairs = [('layer_aie_orig', 0), ('layer_aie_after', 1)], can be None
    """
    # filtered_df = df[(bool(df[f'{model_name}_can_answer']) == True) & (bool(df[f'{model_name}_can_answer']) == False)]
    filtered_df = df[(df[f'{model_name}_can_answer'] == True) & (df[f'{model_name}_can_answer_after'] == 0.0)]

    if columns_to_select != None:
        return filtered_df[columns_to_select]
    else:
        return filtered_df

for dataset in datasets:
    dataset_name = dataset.__class__.__name__
    print("-->dataset_name", dataset_name)
    # filtered_columns = [col for col in dataset.columns if any(substring in col for substring in substrings)]
    # # print(filtered_columns)
    # filtered_dataset = dataset[filtered_columns]
    # # print(filtered_dataset)

    for model_path, model_name in zip(model_paths, model_names):
        print("-->model_name", model_name)
        questions = dataset["question"]
        answers = dataset["answer"]

        answers = dataset["answer"]
        if "Llama-2-13b" in model_name:
            alternative_model_name = "llama-13b"
        if "Mistral" in model_name:
            alternative_model_name = "llama-30b"
        if model_name + "_can_answer" in dataset:
            if_can_answers = dataset[model_name + "_can_answer"]
        elif alternative_model_name + "_can_answer" in dataset:
            if_can_answers = dataset[alternative_model_name + "_can_answer"]
            dataset[model_name + "_can_answer"] = if_can_answers
        else:
            raise Exception("Run dataset.check_if_model_can_answer() first!")


        mt = ModelAndTokenizer(
        model_path + model_name,
        low_cpu_mem_usage=True,
        device='cuda:0'
        )
        mt.model
        print("-->Model loading successfully")

        statements = []
        labels = []
        for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
            start_time = time.time()
            question = row['question']
            expected_answer = row['answer']
            # statement = row['statement']
            if_can_answer = row[model_name + "_can_answer"]

            if bool(if_can_answer) == False:
                print("-->model can't answer such question")
                continue
            prompt_orig = prepare_prompt(question, dataset.truth_instructions[0])

            answer_true = generate_answer(prompt_orig, mt)

            end_time = time.time()
            print("-->time", end_time - start_time)

            lie_instructions = dataset.lie_instructions
            num = random.randint(0, len(lie_instructions) - 1)
            new_prompt = prepare_prompt(question, lie_instructions[num])
            prompt_lie = new_prompt

            answer_lie = generate_answer(prompt_lie, mt)

            if "*out loud*" in answer_lie:
                answer_lie = answer_lie.split("*out loud*")[1].strip()
                if "Question:" in answer_lie:
                    answer_lie = answer_lie.split("Question:")[0].strip()
            # check if the generated answer cover expected_answer
            can_answer_lie = dataset._is_answer_correct(
                answer_lie, expected_answer, question
            )
            print("-->can_answer after lie", can_answer_lie)

            if bool(can_answer_lie) == False:
                statements.append(answer_true)
                labels.append(1)
                statements.append(answer_lie)
                labels.append(0)
        
            # dataset.loc[index, f"{model_name}_false_statement"] = answer
            dataset.loc[index, f"{model_name}_can_answer_after"] = can_answer_lie
            dataset.loc[index, f"{model_name}_answer_true"] = answer_true
            dataset.loc[index, f"{model_name}_answer_lie"] = answer_lie
            

        save_progress = True
        if save_progress:
            # todo use dataset.complete_filename
            print("-->dataset.complete_filename", dataset.complete_filename)
            dataset_name = dataset.__class__.__name__
            dataset.save_processed(None)

        # Define the file path
        csv_file_path = f'./data/raw_questions/{dataset_name}_{model_name}.csv'
        # Open the CSV file for writing
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['statement', 'label'])
            for statement, label in zip(statements, labels):
                writer.writerow([statement, label])
        print(f"CSV file saved at {csv_file_path}")
#  '''       




    # for model_name in model_names:
    #     conditions = {
    #             f"{model_name}_can_answer": True,
    #             f"{model_name}_can_answer_after": False
    #         }
    #     columns_to_select = ['statement', 'false_statement', f"{model_name}_false_statement", f"{model_name}_layer_aie_orig", f"{model_name}_layer_aie_after"]
    #     dataset = filter_dataset(dataset, conditions, None)
        
    #     statements = []
    #     labels = []
    #     for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
    #         statements.append(row['statement'])
    #         labels.append(1)
    #         # statements.append(row['false_statement'])
    #         statements.append(row[f'{model_name}_false_statement'])
    #         labels.append(0)
        
    #     # Define the file path
    #     csv_file_path = f'./data/raw_questions/{dataset_name}_{model_name}.csv'
    #     # Open the CSV file for writing
    #     with open(csv_file_path, mode='w', newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerow(['statement', 'label'])
    #         for statement, label in zip(statements, labels):
    #             writer.writerow([statement, label])

    #     print(f"CSV file saved at {csv_file_path}")
        


