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

from lllm.questions_loaders import Questions1000, WikiData, Commonsense2, Sciq, MathematicalProblems, \
    BBQ, SocialChem, InnodateBias, AutoDAN, GCG, PAP

from bias_detection.TrustGPT.utils.metric.Toxicity import get_toxicity_value

import multiprocessing as mp

template_name = 'llama-2'
conv_template = load_conversation_template(template_name)

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

datasets = [Questions1000(), WikiData(), Commonsense2(), Sciq(), MathematicalProblems(), 
            BBQ(category='gender'), BBQ(category='religion'), BBQ(category='race'), BBQ(category='sexual_orientation'),
            SocialChem(processed_filename='TrustGPT/social-chem-101_1w'),
            AutoDAN(), GCG(), PAP()]

substrings = ['Llama-2-7b', 'Llama-2-13b', 'Llama-3.1', 'Mistral-7B']


for dataset in datasets:
    print("-->", dataset.__class__.__name__)
    filtered_columns = [col for col in dataset.columns if any(substring in col for substring in substrings)]
    print(filtered_columns)
