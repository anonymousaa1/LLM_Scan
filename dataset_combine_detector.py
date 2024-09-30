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


def train_detector(dataset, item=None, save_dir=None):
    if item == None:
        all_causality_effects = [eval(x_item) for x_item in dataset['x']]
    elif isinstance(item, str):
        all_causality_effects = [eval(x_item) for x_item in dataset[item]]
    elif isinstance(item, list):
        all_causality_effects = dataset[item].values.tolist()


    all_labels = dataset['label']

    # training
    print("-->training")
    X_train = np.array(all_causality_effects)
    Y_train = np.array(all_labels)
    print("-->X_train", X_train.shape)
    print("-->Y_train", Y_train.shape)
    
    linear_model_aie = LinearRegression()
    linear_model_aie.fit(X_train, Y_train)
    logistic_model_aie = LogisticRegression()
    logistic_model_aie.fit(X_train, Y_train)
    mlp_regressor = MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', solver='adam', max_iter=500)
    mlp_regressor.fit(X_train, Y_train)
    mlp_classifier = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='adam', max_iter=500)
    mlp_classifier.fit(X_train, Y_train)

    if save_dir != None:
        dump(logistic_model_aie, save_dir + 'logistic_regression_aie.joblib')
        dump(linear_model_aie, save_dir + 'linear_model_aie.joblib')
        # dump(logistic_model_kurt, save_dir + 'logistic_model_kurt.joblib')
        # dump(linear_model_kurt, save_dir + 'linear_model_kurt.joblib')
        dump(mlp_regressor, save_dir + 'mlp_regressor.joblib')
        dump(mlp_classifier, save_dir + 'mlp_classifier.joblib')

    return logistic_model_aie, linear_model_aie, mlp_regressor, mlp_classifier

def evaluate_detector(x, labels, model):

    y_test = np.array(labels)
    x_test = np.array(x)

    y_pred = model.predict(x_test)
    if isinstance(model, LogisticRegression):
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)
        y_pred_proba = model.predict_proba(x_test)[:, 1]  # Probability of positive class
        roc = roc_auc_score(y_test, y_pred_proba)
        print("ROC:", roc)
    elif isinstance(model, LinearRegression):
        y_pred_binary = (y_pred >= 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred_binary)
        print("Accuracy:", accuracy)
        roc = roc_auc_score(y_test, y_pred)
        print("ROC:", roc)
    else:
        y_pred_binary = (y_pred >= 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred_binary)
        print("Accuracy:", accuracy)
        roc = roc_auc_score(y_test, y_pred)
        print("ROC:", roc)
    return accuracy, roc

def evaluate_detector_all(dataset, logistic_model_aie, linear_model_aie, mlp_regressor, mlp_classifier, item=None):
    if item == None:
        all_aies = [eval(x_item) for x_item in dataset['x']]
    elif isinstance(item, str):
        all_aies = [eval(x_item) for x_item in dataset[item]]
    elif isinstance(item, list):
        all_aies = dataset[item].values.tolist()

    all_labels = dataset['label']
    
    print("-->testing")
    X_test = np.array(all_aies)
    Y_test = np.array(all_labels)
    print("-->X_test", X_test.shape)
    print("-->Y_test", Y_test.shape)

    # evaluate on aie value
    print("->logistic_model_aie")
    evaluate_detector(all_aies, all_labels, model=logistic_model_aie)
    print("->linear_model_aie")
    evaluate_detector(all_aies, all_labels, model=linear_model_aie)

    print("->mlp_regressor")
    evaluate_detector(all_aies, all_labels, model=mlp_regressor)
    print("->mlp_classifier")
    evaluate_detector(all_aies, all_labels, model=mlp_classifier)

def evaluate_detector_combine(dataset, model_1, model_2):
    x_1 = [eval(x_item) for x_item in dataset['layer_aie']]
    x_2 = dataset[["mean", "std", "range", "kurtosis", "skewness"]].values.tolist()
    all_labels = dataset['label']

    print("-->testing")
    X_test_1 = np.array(x_1)
    X_test_2 = np.array(x_2)
    Y_test = np.array(all_labels)
    print("-->X_test_1", X_test_1.shape)
    print("-->X_test_2", X_test_2.shape)

    y_pred_1 = model_1.predict(X_test_1)
    y_pred_2 = model_2.predict(X_test_2)

    print("-->y_pred_1", y_pred_1)
    print("-->y_pred_2", y_pred_2)

    y_pred_binary_1 = (y_pred_1 >= 0.5).astype(int)
    accuracy = accuracy_score(Y_test, y_pred_binary_1)
    print("Accuracy:", accuracy)
    roc = roc_auc_score(Y_test, y_pred_1)
    print("ROC:", roc)

    y_pred_binary_2 = (y_pred_2 >= 0.5).astype(int)
    accuracy = accuracy_score(Y_test, y_pred_binary_2)
    print("Accuracy:", accuracy)
    roc = roc_auc_score(Y_test, y_pred_1)
    print("ROC:", roc)

    # Average the probabilities
    y_pred_combine = (y_pred_1 + y_pred_2) / 2
    # y_pred_combine = np.maximum(y_pred_1, y_pred_2)
    combined_predictions = (y_pred_combine > 0.5).astype(int)
    # combined_predictions = (y_pred_binary_1 | y_pred_binary_2).astype(int)
    combined_accuracy = accuracy_score(Y_test, combined_predictions)
    print(f'Combined Accuracy: {combined_accuracy:.4f}')
    
    combined_auc = roc_auc_score(Y_test, y_pred_combine)
    print(f'Combined AUC: {combined_auc:.4f}')


# dataset_names = ["Questions1000", "WikiData", "Sciq", "Commonsense2", "MathematicalProblems", 
#                  "SocialChem", 
#                  "BBQ_gender", "BBQ_religion", "BBQ_race", "BBQ_sexual_orientation"]
# tasks = ['Lies'] * 5 + ['Toxic'] * 1 + ['Bias'] * 4

# dataset_names = ["AutoDAN", "GCG", "PAP"]
# tasks = ['Jailbreak'] * 3 

dataset_names = ["BBQ_gender"]
tasks = ['Bias']

# model_names = ["Llama-2-7b-chat-hf",  "Llama-2-13b-chat-hf", "Meta-Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.2"]
model_name = "Llama-2-7b-chat-hf"

for dataset_name, task in zip(dataset_names, tasks):
    print("-->dataset_name", dataset_name)
    file_name = "data/processed_questions/combined_dataset/" + f"{task}_{dataset_name}_{model_name}.json"
    print("-->file_name", file_name)

    data = pd.read_json(file_name)
    dataset = pd.DataFrame(data)
    dataset = dataset.dropna(subset=['x'])

    train_dataset, test_dataset = train_test_split(dataset, test_size=0.3, random_state=1)  # Ensures reproducibility
    print("Training set size:", len(train_dataset))
    print("Testing set size:", len(test_dataset))

    print("-->dataset", len(dataset))
    test_dataset_specific = dataset.iloc[[314, 1232, 1237]]
    print("-->test_dataset_specific", test_dataset_specific)
    print("-->prompt", test_dataset_specific['prompt'].tolist())
    print("-->label", test_dataset_specific['label'].tolist())

    # detector_saving_dir = saving_dir + "lie-detector/" + dataset_name + "/"
    # if not os.path.exists(detector_saving_dir):
    #     os.makedirs(detector_saving_dir)
    try:
        logistic_model_aie_1, linear_model_aie_1, mlp_regressor_1, mlp_classifier_1 = train_detector(train_dataset, item="layer_aie")
        evaluate_detector_all(dataset=test_dataset,logistic_model_aie=logistic_model_aie_1, 
                          linear_model_aie=linear_model_aie_1,mlp_regressor=mlp_regressor_1, 
                          mlp_classifier=mlp_classifier_1, item="layer_aie")
    except:
        print("-->Fail to train logistic_model_1")

    try:
        logistic_model_aie_2, linear_model_aie_2, mlp_regressor_2, mlp_classifier_2 = train_detector(train_dataset, item=["mean", "std", "range", "kurtosis", "skewness"])
        evaluate_detector_all(dataset=test_dataset,logistic_model_aie=logistic_model_aie_2, 
                          linear_model_aie=linear_model_aie_2,mlp_regressor=mlp_regressor_2, 
                          mlp_classifier=mlp_classifier_2, item=["mean", "std", "range", "kurtosis", "skewness"])
    except:
        print("-->Fail to train logistic_model_2")
    
    # try:
    #     evaluate_detector_combine(test_dataset, mlp_regressor_1, mlp_regressor_2)
    # except:
    #     print("-->Fail to combine")
    print("-->CASE Example")
    evaluate_detector_combine(test_dataset_specific, mlp_regressor_1, mlp_regressor_2)
    



