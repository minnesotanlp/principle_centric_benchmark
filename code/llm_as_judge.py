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

prompt = """-----
Evaluate the input-output pair given a principle! You must score the input-output pair based on this rubric:

Score 0: The principle is not relevant to the response. 
Score 1: The response does not follow the principle at all.
Score 2: The response follows the principle poorly.
Score 3: The response partially follows the principle.
Score 4: The response sufficiently follows the principle.
Score 5: The response correctly and fully follows the principle.

Your response should be in a JSON format of [{"principle_name": "principle_name, "score": "score", "reason": "your reason why you give that score"}. 
-----

"""         
    

def principle_check(task_name, input_file, output_file):
    task2principles = load_extracted_principle_and_definition()  
    # task2principles = load_zeroshot_principle_and_definition() #uncomment this for loading generated principles 
    principles = task2principles[task_name]
    principle2data = load_json(input_file)
    inputs = []
    outputs = []
    raw_answers = []
    principle_list = []
    definitions = []
    for principle, definition in tqdm(principles.items(), total=len(principles)):
    
        principle_text = "##Principle\n{}: {}\n\n".format(principle,definition)
        data = principle2data[principle] 
        
        for key, the_text in data.items():            
            
            input_text = the_text["input"]
            output_text = the_text["output"]
            full_prompt = principle_text + "\n##Input: {}\n\n##Output: {}\n\n".format(input_text, output_text) + prompt +"##Score:\n"

            # print(full_prompt)
            
            answer = chatgpt_prompter(full_prompt)
            # print(answer)
            # quit()
            inputs.append(input_text)
            outputs.append(output_text)
            raw_answers.append(answer)
            principle_list.append(principle)
            definitions.append(definition)
            
    temp_dict = {
        "input": inputs,
        "output": outputs,
        "raw_score": raw_answers,
        "principle": principle_list,
        "definition": definitions
    }
    df = pd.DataFrame(temp_dict)
    df.to_csv(output_file, sep="\t", index=False)

def main():
    task_name = "travel_plan"
    input_file = "../data/gemma_outputs/extracted_principles_{}.json".format(task_name)
    output_file = "scores_{}.tsv".format(task_name)
    principle_check(task_name, output_file)
    

main()