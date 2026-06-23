import pandas as pd 
import os
import openai
from tqdm import tqdm
import random
import time
import json
import re
import ast
from utils import *

openai.api_key = ""

def no_principle_data_generator(task_name, task_description, seed_data, n_shot_instance):

    start_time = time.time()
    no_principle_prompt = """##Task name: {}

##Task description: {}

##Example pairs:
""".format(task_name, task_description)
    
    # print(no_principle_prompt)
    temp_str = ""
    for idx, instance in enumerate(seed_data):        
        new_idx = idx+1
        current_str = """##Input {}: {}

##Output {}: {}
""".format(new_idx, instance["input"], new_idx, instance["output"])
        temp_str += current_str + "\n"
        if idx >= n_shot_instance-1:
            break
    continued_prompt = "-----\nNow come up with 10 input-output pairs for the specified task. Ensure that these new pairs explore topics not addressed in the existing examples. Maintain the same format as the example pairs provided.\n\n"
    
    response = """Your response must be in a JSON format as follows {"1": {"input": "text", "output": "text", "2": ...]
-----
##Response:
"""
    full_prompt = no_principle_prompt + temp_str + continued_prompt + response

    answer = chatgpt_prompter(full_prompt)

    end_time = time.time()
    # Calculate the time difference in seconds
    execution_time_seconds = end_time - start_time
    execution_time_minutes = execution_time_seconds / 60

    print(f"Execution time: {execution_time_minutes:.2f} minutes")
    return answer


def single_principled_data_generator(task_name, task_description, seed_data, n_shot_instance, principle, definition):
    start_time = time.time()
    no_principle_prompt = """##Task name: {}

##Task description: {}

##Example pairs:
""".format(task_name, task_description)
    
    # print(no_principle_prompt)
    temp_str = ""
    for idx, instance in enumerate(seed_data):        
        new_idx = idx+1
        current_str = """##Input {}: {}

##Output {}: {}
""".format(new_idx, instance["input"], new_idx, instance["output"])
        temp_str += current_str + "\n"
        if idx >= n_shot_instance-1:
            break
    principle_prompt = "-----\nMake sure your generated pair aligns with the principle defined below.\n\n##Principle\n{}: {}".format(principle, definition)
    continued_prompt = "\n\nNow come up with 10 input-output pairs for the specified task. Ensure that these new pairs explore topics not addressed in the existing examples. Maintain the same format as the example pairs provided.\n\n"
    
    response = """Your response must be in a JSON format as follows {"1": {"input": "text", "output": "text", "2": ...]
-----
##Response:
"""
    full_prompt = no_principle_prompt + temp_str +  principle_prompt + continued_prompt + response
    answer = chatgpt_prompter(full_prompt)
    # End time
    end_time = time.time()
    # Calculate the time difference in seconds
    execution_time_seconds = end_time - start_time
    execution_time_minutes = execution_time_seconds / 60

    print(f"Execution time: {execution_time_minutes:.2f} minutes")
    return answer

#Baseline data generator (no principles)
def no_principle(output_file):
    task2pairs_biggen = get_task2input_output_biggen()
    task2description_dict = get_task2definition_from_biggen()
    temp_dict = {}
    for task_name in tqdm(task2pairs_biggen.keys(), total=len(task2pairs_biggen)):
        # print(task_name)
        task_description = task2description_dict[task_name]
        # print(task_description)
        n_shot = 10
        response = no_principle_data_generator(task_name, task_description, task2pairs_biggen[task_name], n_shot)
        temp_dict[task_name] = response
    write_json(temp_dict, output_file)

def one_principle(task2principles):
    task2pairs_biggen = get_task2input_output_biggen()
    task2description_dict = get_task2definition_from_biggen()
    temp_dict = {}
    for task_name in tqdm(task2pairs_biggen.keys(), total=len(task2pairs_biggen)):
        task_description = task2description_dict[task_name]        
        n_shot = 10
        response = single_principled_data_generator(task_name, task_description, task2pairs_biggen[task_name], n_shot)
        temp_dict[task_name] = response
    write_json(temp_dict, "biggen_data/data_points_no_principles_10samples.json")

def main():
    # no_principle()
    #Generate benchmark data based on principles!
    task2pairs_biggen = get_task2input_output_biggen()
    task2description_dict = get_task2definition_from_biggen()
    task2principles = load_extracted_principle_and_definition() #replace with load_zeroshot_principle_and_definition() for loading generated principles
    for task_name in tqdm(task2principles.keys(), total=len(task2principles)):
        print(task_name)

        n_shot = 10
        
        extracted_principles = task2principles[task_name]
        task_description = task2description_dict[task_name]

        task2pairs_biggen = get_task2input_output_biggen()
        task2description_dict = get_task2definition_from_biggen()
        task2gen_data = {}
        for principle, definition in tqdm(extracted_principles.items(), total=len(extracted_principles)):
            
            response = single_principled_data_generator(task_name, task_description, task2pairs_biggen[task_name], n_shot, principle, definition)
            
            task2gen_data[principle] = response
            
        write_json(task2gen_data, "extracted_principled_{}_10samples.json".format(task_name))
    
 

main()