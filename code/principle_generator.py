from utils import *
import openai
import pandas as pd
from tqdm import tqdm
import random 
import numpy as np


principle_prompt = """A principle characterizes a specific task. Each task instance consists of a pair: (Input, Output). 

Given a task and its definition, you must generate as many diverse principles as possible. These principles will subsequently be used to generate additional synthetic data (input, output) for the task.

**Guidelines for Generating Principles:**
1. **Naming**:
   - Each principle must have a name consisting of 1 or 2 words only.

2. **Description**:
   - Provide a single-line description for each principle, clearly explaining its relevance to the task.

3. **Uniqueness**:
   - Ensure that all principles are unique and specifically tailored to the task being described.

-----
Example:
Task name: Social Deduction Game
Task description: Persuasive dialogue among multiple players in a social deduction game (Werewolf)
Principle: "Deception Modeling"
Principle definition: "Include scenarios where players intentionally mislead others, paired with annotations indicating when deception occurs."
------
"""

def get_biggen_tasks():
	data_path = "../data/biggen_sample_evals.json"
	with open(data_path, 'r') as file:
		data = json.load(file)
	
	taskname2input_output_pairs = {}
	for key, instance in data.items():
		task = instance["task"]
		capability = instance["capability"]
		the_input = instance["input"]
		the_output = instance["reference_answer"]
		if task not in taskname2input_output_pairs:
			taskname2input_output_pairs[task] = [
				{
					"input": the_input, 
					"output": the_output
				}
			]
		else:
			taskname2input_output_pairs[task].append(
				{
					"input": the_input, 
					"output": the_output
				}
			)

	return taskname2input_output_pairs


def generate_principles(task2input_output, target_task, n_shot_instance):
	task2definition = get_task2definition_from_biggen()
	# print(target_task)
	task_definiton = task2definition[target_task]
	# print(task_definiton)
	instances = task2input_output[target_task]
	format_for_principles = """The format for each principle should be a JSON list as follows:
[{Principle Name} : {A single line describing the generated principle for that task}, ...]"""
	zero_specific = """Now generate as many unique principles as possible for the following task!

##Task name: {}
##Task definition: {}
""".format(target_task.replace("_"," "), task_definiton) + "\n" + format_for_principles + "\n-----\n##New principles:\n"
	
	if n_shot_instance == 0:
		new_prompt = principle_prompt  + zero_specific
	else:
		numbering = []
		temp_str = ""
		cur_list = task2input_output[target_task]
		random.shuffle(cur_list)
		for idx, instance in enumerate(cur_list):
			numbering.append(idx+1)
			new_idx = idx+1
			current_str = """##Input {}: {}

##Output {}: {}
""".format(new_idx, instance["input"], new_idx, instance["output"])
			temp_str += current_str + "\n"
			if idx >= n_shot_instance-1:
				break
		new_prompt = zero_specific + temp_str + "\n-----\n" + format_for_principles + "\n-----\n##New principles:\n"		
	print("PROMPT: ", new_prompt)
	# quit()
	answer = chatgpt_prompter(new_prompt)
	return answer


def main():
	n_shot=0

	task2input_output = get_biggen_tasks()	
	task2new_principles = {}
	for task in tqdm(task2input_output.keys(), total=len(task2input_output)):
		target_task = task
		answer = generate_principles(task2input_output, target_task, n_shot)
		task2new_principles[target_task] = answer

	write_json(task2new_principles, "zeroshot_principles.json".format(n_shot))

main()