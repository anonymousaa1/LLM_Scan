import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import json
import os 
from lllm.questions_loaders import WikiData


def plot_causal_effect(dataset_name, saving_file_name, aie_values, kurt_value=None, analyse_layer=None, target='layer'):
    if target == 'layer':
        seq = aie_values
        kurt = kurt_value
        sns.set_theme()
        plt.figure(figsize=(9, 6))
        sns.scatterplot(x=range(1, len(seq) + 1), y=seq, color='b')

        # dataset_name = dataset.__class__.__name__
        plt.title('Dataset: ' + str(dataset_name))
        # plt.figtext(0.3, 0.03, f'Logits: {logits:.4f}', ha='center', va='center')
        plt.figtext(0.7, 0.03, f'Average Kurtosis Diff: {kurt:.4f}', ha='center', va='center')
        plt.savefig(saving_file_name, bbox_inches="tight")
    elif target == 'neuron':
        sns.set_theme()
        plt.figure(dpi=1000)
        plt.figure(figsize=(9, 6))

        seq = aie_values
        sns.scatterplot(x=range(0, len(seq)), y=seq, color='b', s=20)

        plt.title(dataset_name + ' for Layer: ' + str(analyse_layer))

        averageaie_Range = np.max(seq) - np.min(seq)

        # plt.figtext(0.3, 0, f'Logits: {logits:.4f}', ha='center', va='center')
        plt.figtext(0.3, 0, f'average AIE_Range: {average_aieRange:.4f}, Range of average_AIE: {averageaie_Range:.4f}',
                    ha='center', va='center')
        plt.annotate(f'({np.argmax(seq)},{str(seq[np.argmax(seq)])[:5]})', (np.argmax(seq), seq[np.argmax(seq)]),
                     textcoords="offset points", xytext=(-2, -15), ha='center')

        plt.tight_layout()
        plt.savefig(saving_file_name, bbox_inches="tight")

def get_y1_y2(model_name, dataset_name, task):
    if task == "lie":
        json_file = "outputs_lie/" + model_name + "/layer_AIE_lie.json"
        with open(json_file) as file:
            data = json.load(file)
        y1 = data[dataset_name]['average_AIE_orig']
        y2 = data[dataset_name]['average_AIE_lie']
        return y1, y2
    elif task == 'jailbreak':
        json_file = "outputs_jailbreak/" + model_name + "/layer_AIE_jailbreak.json"
        with open(json_file) as file:
            data = json.load(file)
            # print("-->data", data)
        y1 = data[dataset_name]['average_AIE_non_adv']
        y2 = data[dataset_name]['average_AIE_adv']
        return y1, y2
    elif task == 'toxic':
        json_file = "outputs_toxic/" + model_name + "/layer_AIE_toxic.json"
        with open(json_file) as file:
            data = json.load(file)
        y1 = data[dataset_name]['average_AIE_orig']
        y2 = data[dataset_name]['average_AIE_attemp']
        return y1, y2
    elif task == 'bias':
        json_file = "outputs_bias/" + model_name + "/layer_AIE_bias.json"
        with open(json_file) as file:
            data = json.load(file)
        # y1 = data[dataset_name]['average_AIE_nonstereotype']
        # y2 = data[dataset_name]['average_AIE_stereotype']
        y1 = data[dataset_name]['average_AIE_nonstereotype']
        y2 = data[dataset_name]['average_AIE_stereotype']
        return y1, y2

'''
dataset = WikiData()
print("-->columns", list(dataset.columns))
model_name = "Mistral-7B-Instruct-v0.2"

print(dataset["Mistral-7B-Instruct-v0.2_layer_aie_orig"])

import ast

layer_aie_orig = []
layer_aie_after = []
for index, row in dataset.iterrows():
    
    try:
        layer_aie_orig.append(ast.literal_eval(row["Mistral-7B-Instruct-v0.2_layer_aie_orig"]))
        layer_aie_after.append(ast.literal_eval(row["Mistral-7B-Instruct-v0.2_layer_aie_after"]))
    except:
        continue

print("-->length", len(layer_aie_orig), layer_aie_orig[0], type(layer_aie_orig[0]))
# average_layer_aie_orig = [int(sum(items) / len(items)) for items in zip(*layer_aie_orig)]
# average_layer_aie_after = [int(sum(items) / len(items)) for items in zip(*layer_aie_after)]
average_layer_aie_orig = [sum(items) / len(items) for items in zip(*layer_aie_orig)]
average_layer_aie_after = [sum(items) / len(items) for items in zip(*layer_aie_after)]

print("-->average_layer_aie_orig", average_layer_aie_orig)
print("-->average_layer_aie_after", average_layer_aie_after)
'''

# dataset_name = "MathematicalProblems"
# task = 'lie'

# dataset_name = "PAP"
# task = 'jailbreak'

# dataset_name = "SocialChem"
# task = 'toxic'

dataset_name = "BBQ_sexual_orientation"
task = 'bias'

#llama-2-7b
model_name = "llama-2-7b"
y1, y2 = get_y1_y2(model_name, dataset_name, task)
#llama-2-13b
model_name = "llama-2-13b"
y3, y4 = get_y1_y2(model_name, dataset_name, task)
#llama-3.1
model_name = "llama-3"
y5, y6 = get_y1_y2(model_name, dataset_name, task)
#mistral
model_name = "mistral"
y7, y8 = get_y1_y2(model_name, dataset_name, task)


df_labeled = pd.DataFrame({
    'Value': np.concatenate([y1, y2, y3, y4, y5, y6, y7, y8]),
    'Response': (['llama-2-7b Truth '] * len(y1) + 
                 ['llama-2-7b Lie '] * len(y2) +
                 ['llama-2-13b Truth '] * len(y3) + 
                 ['llama-2-13b Lie '] * len(y4) +
                 ['llama-3.1 Truth '] * len(y5) + 
                 ['llama-3.1 Lie '] * len(y6) +
                 ['Mistral Truth '] * len(y7) + 
                 ['Mistral Lie '] * len(y8))
})

# 全局字体设置
plt.rcParams['font.size'] = 22
plt.rcParams['font.family'] = 'DejaVu Sans'

plt.figure(figsize=(16, 7))
# Violin plot for the two subsets with specified labels
# sns.violinplot(x='Response', y='Value', data=df_labeled, scale='width', inner='box', 
#                palette=['#B6D7A8', '#F2AA84', '#B6D7A8', '#F2AA84', '#B6D7A8', '#F2AA84', '#B6D7A8', '#F2AA84'], linewidth=2)
sns.violinplot(x='Response', y='Value', data=df_labeled, scale='width', inner='box', 
               palette=['#97BDDF', '#F7AB68', '#97BDDF', '#F7AB68', '#97BDDF', '#F7AB68', '#97BDDF', '#F7AB68'], linewidth=2)
# sns.violinplot(x='Response', y='Value', data=df_labeled, scale='width', inner='quartile', palette=['#B6D7A8', '#F2AA84'], linewidth=1)
# plt.title('Distribution of layer ACE for Truth and Lie Responses (Violin Plot)')
plt.ylabel('ACE', )
# plt.grid(True)
# plt.xticks(rotation=45, ha='right', fontsize=12, fontfamily='DejaVu Sans')

# 添加图例
if task == 'lie':
    label1 = 'Truth Response'
    label2 = 'Lie Response'
elif task == 'jailbreak':
    label1 = 'Refusal Response'
    label2 = 'Jailbreak Response'
elif task == 'toxic':
    label1 = 'Normal Response'
    label2 = 'Toxic Response'
elif task == "bias":
    label1 = 'Normal Response'
    label2 = 'Biased Response'
truth_patch = mpatches.Patch(color='#97BDDF', label=label1)  # B6D7A8,  97BDDF
lie_patch = mpatches.Patch(color='#F7AB68', label=label2)  # F2AA84,  F7AB68
plt.legend(handles=[truth_patch, lie_patch], title='Response Type')

plt.tight_layout()
plt.savefig('fig_violin/fig_' + dataset_name + '.pdf')

# saving_path = "./fig_"
# saving_file_name = saving_path + dataset_name + "_nonadv.pdf"
# plot_causal_effect(dataset_name, saving_file_name, y1, 0 , analyse_layer=None, target='layer')
# saving_file_name = saving_path + dataset_name + "_adv.pdf"
# plot_causal_effect(dataset_name, saving_file_name, y2, 0 , analyse_layer=None, target='layer')
