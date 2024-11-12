'''
Author: zhangmengling zhangmengdi1997@126.com
Date: 2024-06-05 13:12:38
LastEditors: zhangmengling zhangmengdi1997@126.com
LastEditTime: 2024-06-05 17:49:27
FilePath: /mengdizhang/LLM-LieDetector/bias-detection/data/TrustGPT/utils/preprocess.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import json
import pandas as pd


def get_txt_file(filename: str) -> list:
    """
    Read a text file and return its content as a list of strings.

    Args:
    filename: str, the name of the text file.

    Returns:
    data: list, the content of the text file as a list of strings.

    Raises:
    AssertionError: If filename is not a string.

    Example:
    get_txt_file("data.txt")
    """

    assert isinstance(filename, str), "filename must be a string."

    with open(filename, 'r') as f:
        data = f.readlines()
        data = [el.strip('\n') for el in data]

    return data


def get_csv_file(filename: str) -> list:
    """
    Read a CSV file and return the values in the 'res' column as a list.

    Args:
    filename: str, the name of the CSV file.

    Returns:
    all_data: list, the values in the 'res' column as a list.

    Raises:
    AssertionError: If filename is not a string or if the file extension is not 'csv'.
    Exception: If the 'res' column is not found in the CSV file.

    Example:
    get_csv_file("data.csv")
    """

    assert isinstance(filename, str), "filename must be a string."
    assert filename.split('.')[-1] == 'csv', "filename must have a '.csv' extension."

    data = pd.read_csv(filename)

    if 'res' not in data.columns:
        raise Exception("no res column!")

    all_data = data['res'].tolist()
    return all_data


def replace_double_underscore(string, replacements):
    result = string
    for replacement in replacements:
        result = result.replace("__", str(replacement), 1)
    return result


def read_configuration(filename: str) -> dict:
    """
    Read a configuration file and return its content as a dictionary.

    Args:
    filename: str, the name of the configuration file.

    Returns:
    config: dict, the content of the configuration file as a dictionary.

    Raises:
    AssertionError: If filename is not a string.

    Example:
    read_configuration("configuration.json")
    """

    assert isinstance(filename, str), "filename must be a string."

    # import os 
    # # Get the current working directory
    # current_directory = os.getcwd()
    # print(f"Current working directory: {current_directory}")

    # current_file = os.path.abspath(__file__)
    # print(f"Current file location: {current_file}")

    with open(filename, 'r') as f:
        # open configuration json file
        with open("bias_detection/TrustGPT/config/configuration.json") as f:   # "../config/configuration.json" or "config/configuration.json"
            config = json.load(f)
    return config