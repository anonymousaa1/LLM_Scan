'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-03-07 15:37:21
LastEditors: zhangmengling zhangmengdi1997@126.com
LastEditTime: 2024-07-30 14:10:18
FilePath: /mengdizhang/LLM-LieDetector/lllm/utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import ast
from time import sleep

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

# Maximum number of tokens that the openai api allows me to request per minute
RATE_LIMIT = 250000

# To avoid rate limits, we use exponential backoff where we wait longer and longer
# between requests whenever we hit a rate limit. Explanation can be found here:
# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
# I'm using default parameters here, I don't know if something else might be
# better.
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


# Define a function that adds a delay to a Completion API call
def delayed_completion_with_backoff(delay_in_seconds: float = 1, **kwargs):
    """Delay a completion by a specified amount of time."""

    # Sleep for the delay
    sleep(delay_in_seconds)

    # Call the Completion API and return the result
    return completion_with_backoff(**kwargs)


def completion_create_retry(*args, sleep_time=5, **kwargs):
    """A wrapper around openai.Completion.create that retries the request if it fails for any reason."""

    if 'vicuna' in kwargs['model'].lower() or 'alpaca' in kwargs['model'].lower():
        if type(kwargs['prompt'][0]) == list:
            prompts = [prompt[0] for prompt in kwargs['prompt']]
        else:
            prompts = kwargs['prompt']
        return kwargs['endpoint'](prompts, **kwargs)
    elif 'llama' in kwargs['model'].lower(): 
        if type(kwargs['prompt'][0]) == list:
            prompts = [prompt[0] for prompt in kwargs['prompt']]
        else:
            prompts = kwargs['prompt']
        
        device = 'cuda:0'
        input_ids = kwargs['tokenizer'](prompts, padding=True, return_tensors="pt")
        input_ids['input_ids'] = input_ids['input_ids'].to(device)
        input_ids['attention_mask'] = input_ids['attention_mask'].to(device)
        num_input_tokens = input_ids['input_ids'].shape[1]
        outputs = kwargs['endpoint'].generate(input_ids['input_ids'], attention_mask=input_ids['attention_mask'].half(),
                                         max_new_tokens=kwargs['max_new_tokens'], do_sample=False, pad_token_id=kwargs['tokenizer'].pad_token_id)
        generation = kwargs['tokenizer'].batch_decode(outputs[:, num_input_tokens:], skip_special_tokens=True)
        return generation
    elif 'mistral' in kwargs['model'].lower():
        if type(kwargs['prompt'][0]) == list:
            prompts = [prompt[0] for prompt in kwargs['prompt']]
        else:
            prompts = kwargs['prompt']

        device = 'cuda:0'
        input_ids = kwargs['tokenizer'](prompts, padding=True, return_tensors="pt")
        input_ids['input_ids'] = input_ids['input_ids'].to(device)
        input_ids['attention_mask'] = input_ids['attention_mask'].to(device)
        num_input_tokens = input_ids['input_ids'].shape[1]
        outputs = kwargs['endpoint'].generate(input_ids['input_ids'], attention_mask=input_ids['attention_mask'].half(),
                                         max_new_tokens=kwargs['max_new_tokens'], do_sample=False, pad_token_id=kwargs['tokenizer'].pad_token_id)
        generation = kwargs['tokenizer'].batch_decode(outputs[:, num_input_tokens:], skip_special_tokens=True)
        return generation
    else:
        while True:
            try:
                return openai.Completion.create(*args, **kwargs)
            except:
                sleep(sleep_time)
