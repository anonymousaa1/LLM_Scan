'''
Author: zhangmengling zhangmengdi1997@126.com
Date: 2024-07-23 20:03:04
LastEditors: zhangmengling zhangmengdi1997@126.com
LastEditTime: 2024-07-23 20:10:43
FilePath: /mengdizhang/LLM_probing_Causality/data/processed_questions/dataset_with_prompt/check.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import json
import pandas as pd

json_file = "MathematicalProblems.json"
with open(json_file) as file:
    data = json.load(file)

dataset = pd.DataFrame(data)

print(dataset)
print(dataset.columns)

for index, row in dataset.iterrows():
    can_answer = row['Llama-2-7b-chat-hf_can_answer']
    can_answer_after = row['Llama-2-7b-chat-hf_can_answer_after']
    print(f"can_answer:{can_answer}, can_answer_after:{can_answer_after}")