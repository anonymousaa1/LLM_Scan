'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-03-20 13:10:39
LastEditors: zhangmengling zhangmengdi1997@126.com
LastEditTime: 2024-07-23 17:12:05
FilePath: /mengdizhang/LLM-LieDetector/utils/utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from fastchat.model import get_conversation_template


def load_conversation_template(template_name):
    conv_template = get_conversation_template(template_name)
    if conv_template.name == 'zero_shot':
        conv_template.roles = tuple(['### ' + r for r in conv_template.roles])
        conv_template.sep = '\n'
    elif conv_template.name == 'llama-2':
        conv_template.sep2 = conv_template.sep2.strip()
    
    return conv_template


def generate_input(conv_template,prompt,adv_suffix=None):   
    conv_template.messages = []
    if adv_suffix is not None:
        conv_template.append_message(conv_template.roles[0], f"{prompt} {adv_suffix}")
        conv_template.append_message(conv_template.roles[1], None)
        result = conv_template.get_prompt()
    else:
        conv_template.append_message(conv_template.roles[0], f"{prompt}.")
        conv_template.append_message(conv_template.roles[1], None)
        result = conv_template.get_prompt() + " "
    return result

def prepare_prompt(question, instruction=None):
    if instruction == None:
        prompt = "Question: " + question + "\n" + "Answer:"
    else:
        prompt = instruction + "\n" + "Question: " + question + "\n" + "Answer: "
    return prompt

def prepare_prompt_choice(question, choices, instruction=None):
    if instruction == None:
        question_prompt = "Question: " + question + "\n"
        choices_prompt = ''.join([str(i) + ") " + str(choices[i]) + "\n" for i in range(0, len(choices))])
        prompt = question_prompt + choices_prompt + "\n" + "Answer:"
    else:
        question_prompt = instruction + "\n" + "Question: " + question + "\n"
        choices_prompt = ''.join([str(i) +  ") " + str(choices[i]) + "\n" for i in range(0, len(choices))])
        prompt = question_prompt + choices_prompt + "\n" + "Answer:"
    return prompt

def prepare_prompt_completion(temp, values, instruction=None):
    """
    temp:, eg., "Say something {} when {}"
    values: e.g., [prompt_type, question]
    """
    prompt = temp.format(*values) + "\n" + "Example: "
    return prompt