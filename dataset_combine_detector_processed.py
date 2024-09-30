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
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, log_loss
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


def train_detector_prompt(dataset, item=None, save_dir=None):
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

def train_detector(dataset, model_name, task, save_dir=None, target='layer', lie_instruction_num='random'):
    '''
    sample_split_prop = 0.7  # the proportion of training and testing data = 0.7/0.3
    save_dir = "outputs/llama-2-7b/lie-detector/"
    '''
    if target == 'layer':
        if f"{model_name}_layer_aie_orig" in dataset or f"{model_name}_layer_aie" in dataset:
            all_causality_effects, all_labels = get_X_Y_from_dataset_with_condition(dataset, model_name, task, target)
        else:
            # print("-->existing columns", list(dataset.columns))
            raise KeyError()
    elif target == 'prompt':
        if f"{model_name}_prompt_aie_orig" in dataset or f"{model_name}_prompt_aie" in dataset:
            all_causality_effects, all_labels = get_X_Y_from_dataset_with_condition(dataset, model_name, task, target)
        else:
            # print("-->existing columns", list(dataset.columns))
            raise KeyError()
    else:
        raise KeyError("-->Invalid target", target)


    # for neuron/layer get aie from dataset
    # all_causality_effects, all_labels = get_X_Y_from_dataset(dataset, model_name, target=target)
    # for layer
    # all_causality_effects, all_kurt, all_labels = get_aie_kurt(dataset, model_name, target=target)

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


def get_X_Y_from_dataset_with_condition(dataset, model_name, task, target='layer'):
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

        # query_str = ' & '.join([f'({col} == {val})' if isinstance(val, bool) else f'({col} == "{val}")' for col, val in conditions.items()])
        # filtered_df = df.query(query_str, engine='python')
        if columns_to_select != None:
            return filtered_df[columns_to_select]
        else:
            return filtered_df

    if target == 'layer':
        if task  == 'lie':
            print("-->dataset before filter", len(dataset))
            conditions = {
                f"{model_name}_can_answer": True,
                f"{model_name}_can_answer_after": False
            }
            columns_to_select = [f"{model_name}_layer_aie_orig", f"{model_name}_layer_aie_after"]
            dataset = filter_dataset(dataset, conditions, columns_to_select)
            print("-->filtered dataset", len(dataset))
            all_aies = []
            all_labels = []
            for index, row in dataset.iterrows():
                all_aies.append(row[f"{model_name}_layer_aie_orig"])
                all_labels.append(0)
                all_aies.append(row[f"{model_name}_layer_aie_after"])
                all_labels.append(1)
            all_aies = [json.loads(aie) for aie in all_aies]
        elif task == 'toxic':
            toxicity_threshold = 0.5
            print("-->dataset before filter", len(dataset))
            dataset = dataset[
                (dataset[f"{model_name}_toxicity_value_orig"] <= toxicity_threshold) &
                (dataset[f"{model_name}_toxicity_value_after"] > toxicity_threshold)
            ]
            print("-->filtered dataset", len(dataset))
            all_aies = []
            all_labels = []
            for index, row in dataset.iterrows():
                all_aies.append(row[f"{model_name}_layer_aie_orig"])
                all_labels.append(0)
                all_aies.append(row[f"{model_name}_layer_aie_after"])
                all_labels.append(1)
            all_aies = [json.loads(aie) for aie in all_aies]
        elif task == 'bias':
            all_aies = dataset[f"{model_name}_layer_aie"].tolist()
            all_aies = [json.loads(aie) if isinstance(aie, str)==True else aie for aie in all_aies]
            is_stereotype = dataset[f"{model_name}_is_stereotype"]
            # all_labels = [0 if is_c == True else 1 for is_c in is_stereotype] # mistake
            # all_labels = [1 if is_c == True else 0 for is_c in is_stereotype]
            is_correct = dataset[f"{model_name}_is_correct"]
            all_labels = [1 if is_c == True else 0 for is_c in is_correct]
        elif task == 'jailbreak':
            dataset = dataset[dataset[f"{model_name}_prompt_aie"].notnull()]

            all_aies = dataset[f"{model_name}_layer_aie"].tolist()
            all_aies = [json.loads(aie) if isinstance(aie, str)==True else aie for aie in all_aies]
            labels = dataset['label']
            all_labels = [1 if is_adv == 'adv_data' else 0 for is_adv in labels]
    elif target == 'prompt':
        if task  == 'lie':
            print("-->dataset before filter", len(dataset))
            conditions = {
                f"{model_name}_can_answer": True,
                f"{model_name}_can_answer_after": False
            }
            columns_to_select = [f"{model_name}_prompt_aie_orig", f"{model_name}_prompt_aie_after"]
            dataset = filter_dataset(dataset, conditions, columns_to_select)
            print("-->filtered dataset", len(dataset))
            all_aies = []
            all_labels = []
            for index, row in dataset.iterrows():
                all_aies.append(row[f"{model_name}_prompt_aie_orig"])
                all_labels.append(0)
                all_aies.append(row[f"{model_name}_prompt_aie_after"])
                all_labels.append(1)
            all_aies = [json.loads(aie) for aie in all_aies]
        elif task == 'toxic':
            toxicity_threshold = 0.5
            print("-->dataset before filter", len(dataset))
            dataset = dataset[
                (dataset[f"{model_name}_toxicity_value_orig"] <= toxicity_threshold) &
                (dataset[f"{model_name}_toxicity_value_after"] > toxicity_threshold)
            ]
            print("-->filtered dataset", len(dataset))
            all_aies = []
            all_labels = []
            for index, row in dataset.iterrows():
                all_aies.append(row[f"{model_name}_prompt_aie_orig"])
                all_labels.append(0)
                all_aies.append(row[f"{model_name}_prompt_aie_after"])
                all_labels.append(1)
            all_aies = [json.loads(aie) for aie in all_aies]
        elif task == 'bias':
            all_aies = dataset[f"{model_name}_prompt_aie"].tolist()
            all_aies = [json.loads(aie) if isinstance(aie, str)==True else aie for aie in all_aies]
            is_stereotype = dataset[f"{model_name}_is_stereotype"]
            # all_labels = [0 if is_c == True else 1 for is_c in is_stereotype] # mistake
            # all_labels = [1 if is_c == True else 0 for is_c in is_stereotype]
            is_correct = dataset[f"{model_name}_is_correct"]
            all_labels = [1 if is_c == True else 0 for is_c in is_correct]
        elif task == 'jailbreak':
            dataset = dataset[dataset[f"{model_name}_prompt_aie"].notnull()]

            all_aies = dataset[f"{model_name}_prompt_aie"].tolist()
            all_aies = [json.loads(aie) if isinstance(aie, str)==True else aie for aie in all_aies]
            print(type(all_aies[0][0]))
            labels = dataset['label']
            all_labels = [1 if is_adv == 'adv_data' else 0 for is_adv in labels]
    else:
        raise KeyError("-->Invalid target", target)
    # print("-->all_labels", all_labels)
    X = all_aies
    Y = all_labels
    return X, Y

def evaluate_detector_all(dataset, model_name, task, logistic_model_aie, linear_model_aie, mlp_regressor, mlp_classifier, target='layer'):

    if f"{model_name}_layer_aie_orig" in dataset or f"{model_name}_layer_aie" in dataset:
            # all_causality_effects, all_labels = get_X_Y_from_dataset(dataset, model_name, target=target)
        all_aies, all_labels = get_X_Y_from_dataset_with_condition(dataset, model_name, task, target)
    else:
        raise KeyError("")

    # for layer
    # all_causality_effects, all_kurt, all_labels = get_aie_kurt(dataset, model_name, target=target)

    # training
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

def analyse_causality_layer(mt, model_name, dataset, target, saving_dir):
    # -------------------- train & evaluate detector --------------------
    orig_dataset = dataset
    train_df, test_df = train_test_split(dataset, test_size=0.3, random_state=1)  # Ensures reproducibility
    print("Training set size:", len(train_df))
    print("Testing set size:", len(test_df))
    train_dataset = dataset.reset_df(train_df)
    dataset = orig_dataset
    test_dataset = dataset.reset_df(test_df)
    print("train_dataset:", len(train_dataset))
    print("test_dataset:", len(test_dataset))

    detector_saving_dir = saving_dir + "toxic-detector/" + dataset_name + "/"
    if not os.path.exists(detector_saving_dir):
            os.makedirs(detector_saving_dir)
        # training
    logistic_model_aie, linear_model_aie, mlp_regressor, mlp_classifier = train_detector(train_dataset,
                                                                                        model_name,
                                                                                        task,
                                                                                        save_dir=detector_saving_dir,
                                                                                        target=target)
                                                                                        
    # testing
    evaluate_detector_all(dataset=test_dataset, model_name=model_name, task=task,
                                  logistic_model_aie=logistic_model_aie, linear_model_aie=linear_model_aie,
                                  mlp_regressor=mlp_regressor, mlp_classifier=mlp_classifier,
                                  target=target)
    
def evaluate_detector_combine(dataset, model_name, task, model_1, model_2):
            
    x_1, all_labels = get_X_Y_from_dataset_with_condition(dataset, model_name, task, 'layer')
    x_2, all_labels_2 = get_X_Y_from_dataset_with_condition(dataset, model_name, task, 'prompt')
    
    print('-->check if all_labels_2 equals all_labels')
    print(all_labels == all_labels_2)  # Output: True

    print("-->testing")
    X_test_1 = np.array(x_1)
    X_test_2 = np.array(x_2)
    Y_test = np.array(all_labels)
    print("-->X_test_1", X_test_1.shape)
    print("-->X_test_2", X_test_2.shape)

    y_pred_1 = model_1.predict(X_test_1)
    y_pred_2 = model_2.predict(X_test_2)

    # print("-->y_pred_1", y_pred_1)
    # print("-->y_pred_2", y_pred_2)

    y_pred_binary_1 = (y_pred_1 >= 0.5).astype(int)
    accuracy = accuracy_score(Y_test, y_pred_binary_1)
    print("Accuracy:", accuracy)
    roc = roc_auc_score(Y_test, y_pred_1)
    print("ROC:", roc)

    y_pred_binary_2 = (y_pred_2 >= 0.5).astype(int)
    accuracy = accuracy_score(Y_test, y_pred_binary_2)
    print("Accuracy:", accuracy)
    roc = roc_auc_score(Y_test, y_pred_2)
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


if __name__ == '__main__':

    # model_name = "Llama-2-7b-chat-hf"
    # task = 'lie'
    # target = 'layer'

    # dataset = Questions1000()
    # train_df, test_df = train_test_split(dataset, test_size=0.3, random_state=1)  # Ensures reproducibility
    # print("Training set size:", len(train_df))
    # print("Testing set size:", len(test_df))
    # train_dataset = dataset.reset_df(train_df)
    # dataset = Questions1000()
    # test_dataset = dataset.reset_df(test_df)
    # print("train_dataset:", len(train_dataset))
    # print("test_dataset:", len(test_dataset))

           
    # logistic_model_aie1, linear_model_aie1, mlp_regressor1, mlp_classifier1 = train_detector(train_dataset,
    #                                                                                              model_name,
    #                                                                                              task, 
    #                                                                                              save_dir=None,  # save_dir=detector_saving_dir,
    #                                                                                              target='layer',
    #                                                                                              lie_instruction_num='random')
    # evaluate_detector_all(dataset=test_dataset, model_name=model_name, task=task,
    #                               logistic_model_aie=logistic_model_aie1, linear_model_aie=linear_model_aie1,
    #                               mlp_regressor=mlp_regressor1, mlp_classifier=mlp_classifier1,
    #                               target='layer')

    # logistic_model_aie2, linear_model_aie2, mlp_regressor2, mlp_classifier2 = train_detector(train_dataset,
    #                                                                                              model_name,
    #                                                                                              task, 
    #                                                                                              save_dir=None,  # save_dir=detector_saving_dir,
    #                                                                                              target='prompt',
    #                                                                                              lie_instruction_num='random')
    # evaluate_detector_all(dataset=test_dataset, model_name=model_name, task=task,
    #                               logistic_model_aie=logistic_model_aie2, linear_model_aie=linear_model_aie2,
    #                               mlp_regressor=mlp_regressor2, mlp_classifier=mlp_classifier2,
    #                               target='prompt')

    # evaluate_detector_combine(test_dataset, model_name, task, mlp_regressor1, mlp_regressor2)





    # dataset_names = ["Questions1000", "WikiData", "Sciq", "Commonsense2", "MathematicalProblems", 
    #                  "SocialChem", 
    #                  "BBQ_gender", "BBQ_religion", "BBQ_race", "BBQ_sexual_orientation"]
    # datasets = ["Questions1000()", "WikiData()", "Sciq()", "Commonsense2()", "MathematicalProblems()", 
    #             "SocialChem(processed_filename='TrustGPT/social-chem-101_1w')", 
    #             "BBQ(category='gender')", "BBQ(category='religion')", "BBQ(category='race')", "BBQ(category='sexual_orientation')"]
    # tasks = ['lie'] * 5 + ['toxic'] * 1 + ['bias'] * 4

    datasets = ["AutoDAN()", "GCG()", "PAP()"]
    tasks = ['jailbreak'] * 3 

    datasets = ["BBQ(category='race')"]
    tasks = ['bias']

    # datasets = ["BBQ(category='gender')", "BBQ(category='religion')", "BBQ(category='race')", "BBQ(category='sexual_orientation')"]
    # tasks = ['bias'] * 4

    # model_names = ["Llama-2-7b-chat-hf",  "Llama-2-13b-chat-hf", "Meta-Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.2"]
    model_name =  "Meta-Llama-3.1-8B-Instruct"

    for dataset_str, task in zip(datasets, tasks):
        dataset = eval(dataset_str)

        dataset_name = dataset.__class__.__name__
        print("-->dataset_name", dataset_name)
        print("-->columns", list(dataset.columns))
        # '''
        train_df, test_df = train_test_split(dataset, test_size=0.3, random_state=1)  # Ensures reproducibility
        print("Training set size:", len(train_df))
        print("Testing set size:", len(test_df))
        train_dataset = dataset.reset_df(train_df)

        dataset = eval(dataset_str)
        test_dataset = dataset.reset_df(test_df)
        print("train_dataset:", len(train_dataset))
        print("test_dataset:", len(test_dataset))

        # print("-->columns", list(dataset.columns))

           
        logistic_model_aie1, linear_model_aie1, mlp_regressor1, mlp_classifier1 = train_detector(train_dataset,
                                                                                                 model_name,
                                                                                                 task, 
                                                                                                 save_dir=None,  # save_dir=detector_saving_dir,
                                                                                                 target='layer',
                                                                                                 lie_instruction_num='random')
        evaluate_detector_all(dataset=test_dataset, model_name=model_name, task=task,
                                  logistic_model_aie=logistic_model_aie1, linear_model_aie=linear_model_aie1,
                                  mlp_regressor=mlp_regressor1, mlp_classifier=mlp_classifier1,
                                  target='layer')

        logistic_model_aie2, linear_model_aie2, mlp_regressor2, mlp_classifier2 = train_detector(train_dataset,
                                                                                                 model_name,
                                                                                                 task, 
                                                                                                 save_dir=None,  # save_dir=detector_saving_dir,
                                                                                                 target='prompt',
                                                                                                 lie_instruction_num='random')
        evaluate_detector_all(dataset=test_dataset, model_name=model_name, task=task,
                                  logistic_model_aie=logistic_model_aie2, linear_model_aie=linear_model_aie2,
                                  mlp_regressor=mlp_regressor2, mlp_classifier=mlp_classifier2,
                                  target='prompt')

        evaluate_detector_combine(test_dataset, model_name, task, mlp_regressor1, mlp_regressor2)
        # '''



