from lllm.questions_loaders import *
from utils.modelUtils import *
from utils.utils import *
import random

def filter_dataset(df, conditions, columns_to_select=None):
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

orig_dataset = WikiData()
dataset_name = orig_dataset.__class__.__name__
model_name = "Llama-2-7b-chat-hf"
# model_name = "Llama-2-13b-chat-hf"
# model_name = "Meta-Llama-3.1-8B-Instruct"
# model_name = "Mistral-7B-Instruct-v0.2"

print("-->columns", list(orig_dataset.columns))

print("-->dataset before filter", len(orig_dataset))
conditions = {
    f"{model_name}_can_answer": True,
    f"{model_name}_can_answer_after": False
}
columns_to_select = [f"{model_name}_layer_aie_orig", f"{model_name}_layer_aie_after"]
dataset = filter_dataset(orig_dataset, conditions)
print("-->filtered dataset", len(dataset))

lie_instructions = orig_dataset.lie_instructions
prompt_orig =[]
prompt_after = []
for index, row in dataset.iterrows():
    question = row['question']
    prompt_orig.append(question)
    num = random.randint(0, len(lie_instructions) - 1)
    prompt_after.append(lie_instructions[num] + ' ' + question)


import json
saving_filename = f'{dataset_name}_{model_name}.json'
with open(saving_filename, 'w') as file:
    data = {"prompt_orig": prompt_orig, "prompt_after": prompt_after}
    json.dump(data, file)



# model_name = "/common2/public/LLAMA2-HF/Llama-2-7b-chat-hf"
# lie_instruction_num = "random"

# questions = dataset["question"]
# answers = dataset["answer"]
# if_can_answers = dataset[model_name + "_can_answer"]

# all_layerAIE_orig = []
# all_layerAIE_lie = []
# all_layerAIE_lie_attempt = []
# all_kurt_orig = []
# all_kurt_lie = []
# all_kurt_lie_attempt = []

# success_lie_instruct = 0
# attempt_num = 0
# for k, question in enumerate(tqdm(questions)):  # for question in tqdm(questions):
#     expected_answer = answers[k]
#     if_can_answer = if_can_answers[k]
#     if bool(if_can_answer) == False:
#         print("-->model can't answer such question")
#         continue
#     attempt_num += 1

#     prompt_orig = prepare_prompt(question, dataset.truth_instructions[0])
#     ########## todo ##########
#     answer = Generate_model_answer(prompt_orig)  # example of generating model answer function
#     # get AIE for prompt_orig
#     # if you try other model, and there is no dataset[model_name + "_can_answer"] in the original dataset, you can do:
#     # can_answer_orig = dataset._is_answer_correct(
#     #     answer, expected_answer, question
#     # )
#     if can_answer_orig == True:
#         # model can answer
#         # ...
#     else:
#         continue

#     lie_instructions = dataset.lie_instructions
#     if isinstance(lie_instruction_num, int):
#         num = lie_instruction_num
#     elif lie_instruction_num == "random":
#         num = random.randint(0, len(lie_instructions) - 1)
#     if "*out loud*" in lie_instructions[num]:
#         new_prompt = prepare_prompt(question, lie_instructions[num])
#         prompt_lie = new_prompt
#     else:
#         prompt_lie = generate_input(conv_template, lie_instructions[num] + question)

#     ########## todo ##########
#     answer = Generate_model_answer(prompt_lie) # example of generating model answer function
#     # get AIE for prompt_orig
#     # ...

#     if "*out loud*" in answer:
#         answer = answer.split("*out loud*")[1].strip()
#         if "Question:" in answer:
#             answer = answer.split("Question:")[0].strip()
#     # check if the generated answer cover expected_answer
#     can_answer_lie = dataset._is_answer_correct(
#         answer, expected_answer, question
#     )
#     print("-->expected_answer", expected_answer)
#     print("-->can_answer after lie", can_answer_lie)
#     # if bool(can_answer_lie) == True:
#     # print("-->can_answer after lie", can_answer_lie)
#     # print("-->expected_answer", expected_answer)
#     # print("-->answer", answer)