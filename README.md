# LLMScan: Causal Scan for LLM Misbehavior Detection 
This repository is to scan LLM's "brain" and detect LLM's misbehavior based on causality analysis. 

## Abstract

Despite the success of Large Language Models (LLMs) across various fields, their potential to generate untruthful, biased and harmful responses poses significant risks, particularly in critical applications. This highlights the urgent need for systematic methods to detect and prevent such misbehavior. While existing approaches target specific issues such as harmful responses, this work introduces LLMScan, an innovative LLM monitoring technique based on causality analysis, offering a comprehensive solution.LLMScan systematically monitors the inner workings of an LLM through the lens of causal inference, operating on the premise that the LLM's `brain' behaves differently when misbehaving. By analyzing the causal contributions of the LLM's input tokens and transformer layers, LLMScan effectively detects misbehavior. Extensive experiments across various tasks and models reveal clear distinctions in the causal distributions between normal behavior and misbehavior, enabling the development of accurate, lightweight detectors for a variety of misbehavior detection tasks.

## Structure of this repository:

- `data` contains the raw datasets and processed dataset with CE informations for 4 detection tasks. `data/raw_questions` contains the datasets in their original format, while `data/processed_questions` contains the datasets transformed to a common format. (the dataset loading code is at file lllm/questions_loaders.py)
- `lllm`, `utils`: contains source code. 
- `public_fun`: contains the source code running LLMScan (CE generation and detector trianing/evaluation). In specifically, `public_fun/causality_analysis.py` contains the code for scanning model layers and generating layer-level causal effects, `public_fun/causality_analysis.py` contains the code for generating model token-level causal effects and the detector training is executed at `public_fun/causality_analysis_combine.py` which contains the code for training, evaluating our LLMScan detectors. 
- `figs`: contain analyzing figures present in the paper, e.g., PCA, Violin Figures and Causal Maps

`public_fun/paramters.json`

## Setup

The code was developed with Python 3.8. To install dependencies:
```bash
pip install -r requirements.txt
```

## Model
All pre-trained models are loaded from HuggingFace.
```bash
# llama-2-7b
"model_path": "meta-llama/",
"model_name": "Llama-2-7b-chat-hf"

# llama-2-13b
"model_path": "meta-llama/",
"model_name": "Llama-2-13b-chat-hf"

# llama-3.1
"model_path": "meta-llama/",
"model_name": "Meta-Llama-3.1-8B-Instruct"

# Mistral
"model_path": "mistralai/",
"model_name": "Mistral-7B-Instruct-v0.2"
```

## Reproducibility Experiment
```bash
# generating layer-level ce (remember to set the parameter 'save_progress' as True to save all causal effects results in processed_dataset files)
python public_func/causality_analysis.py --model_path "meta-llama/" --model_name "Llama-2-7b-chat-hf" --task "lie" --dataset "Questions1000()" --saving_dir "outputs_lie/llama-2-7b/"
# or you can directly run: 
python public_func/causality_analysis.py   # then the parameters are loaded from file public/parameters.json

# generating token-level ce 
python public_func/causality_analysis_prompt.py

# train and evaluate the detector
python public_func/causality_analysis_combine.py
```

## Ethics Statement:
This research does not involve human subjects or sensitive data. All datasets used in the experiments are publicly available, and no personal or identifiable information is included. We have taken care to ensure that the methodologies employed do not introduce harmful insights or reinforce any forms of bias or discrimination. The models were designed and tested with fairness in mind, and no conflicts of interest or sponsorship concerns are present.

## Reproducibility Statement:
To ensure the reproducibility of our results, we have provided detailed descriptions of the models, datasets, and experimental setups in the main paper and supplementary materials. All theoretical assumptions are clearly outlined, and complete proofs of our claims are included in the appendix. Additionally, we have provided anonymous downloadable source code and documentation as supplementary materials for replicating our experiments.
