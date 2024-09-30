# LLMScan: Causal Scan for LLM Misbehavior Detection 
This repository is to scan LLM's "brain" and detect LLM's misbehavior based on causality analysis. 

## Abstract

Despite the success of Large Language Models (LLMs) across various fields, their potential to generate untruthful, biased and harmful responses poses significant risks, particularly in critical applications. This highlights the urgent need for systematic methods to detect and prevent such misbehavior. While existing approaches target specific issues such as harmful responses, this work introduces LLMScan, an innovative LLM monitoring technique based on causality analysis, offering a comprehensive solution.LLMScan systematically monitors the inner workings of an LLM through the lens of causal inference, operating on the premise that the LLM's `brain' behaves differently when misbehaving. By analyzing the causal contributions of the LLM's input tokens and transformer layers, LLMScan effectively detects misbehavior. Extensive experiments across various tasks and models reveal clear distinctions in the causal distributions between normal behavior and misbehavior, enabling the development of accurate, lightweight detectors for a variety of misbehavior detection tasks.

## Structure of this repository:

- `data` contains the raw datasets and processed dataset with CE informations for 4 detection tasks. `data/raw_questions` contains the datasets in their original format, while `data/processed_questions` contains the datasets transformed to a common format. (the dataset loading code is at file lllm/questions_loaders.py)
- `lllm`, `utils`: contains source code. 
- `public_fun`: contains the source code running LLMScan (CE generation and detector trianing/evaluation). In specifically, `public_fun/causality_analysis.py` contains the code for scanning model layers and generating layer-level causal effects, `public_fun/causality_analysis.py` contains the code for generating model token-level causal effects and the detector training is executed at `public_fun/causality_analysis_combine.py` which contains the code for training, evaluating our LLMScan detectors. 
- `figs`: all analyzing figures, e.g., PCA, Violin Figures and Causal Maps

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

## Example Experiment
```bash
# generating layer-level ce (remember to set the save_progress as True to save all causal effects results in processed_dataset files)
python public_func/causality_analysis.py --model_path "meta-llama/" --model_name "Llama-2-7b-chat-hf" --task "lie" --dataset "Questions1000()" --saving_dir "outputs_lie/llama-2-7b/"
# or you can directly run: 
python public_func/causality_analysis.py   # then the parameters are loaded from file public/parameters.json

# generating token-level ce 
python public_func/causality_analysis_prompt.py

# train and evaluate the detector
python public_func/causality_analysis_combine.py
```





### Other files:
- `lllm` contains additional utilities that are used throughout.
- `imgs` contain a few images present in the paper and a notebook to generate them
- `other` contains utility notebooks to explore the model answers when instructed to lie and to add and test elicitation questions.   


## Practicalities

To use this code, create a clean `Python` environment and then run 

```pip install -r requirements.txt```
```pip install -r requirements_casper.txt```


To run experiments with the open-source models, you need access to a computing cluster with GPUs and to install the [`deepspeed_llama`](https://github.com/LoryPack/deepspeed_llama) repository on that cluster. You'll need to change the source code of that repository to point to the cluster directory where the weights for the open-source models are stored. `experiments_alpaca_vicuna` and `finetuning/llama` contain a few `*.sh` example scripts for clusters using `slurm`.
There are also a few other things that need to be changed in `lllm/llama_utils.py` according to the paths of your cluster. Moreover, `finetuning/llama/llama_ft_folder.json` maps the different fine-tuning setups for Llama to a specific path on the cluster we used, so this needs to be changed too. 

Finally, to run experiments on the OpenAI models, you'll need to store your [OpenAI API key](https://platform.openai.com/account/api-keys) in a `.env` file in the root of this directory, with the format: 

```OPENAI_API_KEY=sk-<your key>```

Running experiments with the OpenAI API will incur a monetary cost. Some of our experiments are extensive and, as such, the costs will be substantial. However, our results are already stored in this repository and, by default, most of our code will load them instead of querying the API. Of course, you can overwrite our results by specifying the corresponding argument to the various functions and methods.