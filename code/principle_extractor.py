from utils import *
import openai
import pandas as pd
from tqdm import tqdm


your_task = """
An evaluation sentence is evaluating the quality of the given output. Your task is to extract only the principle of a good output according to the evaluation sentence. Your answer must include:

* A short name (1–2 words) for each principle
* A definition of the principle

A principle must not be too specific but not too general either. Principles must be distinct from each other.

Your answer must be in the following json format:
[
	{
		"principle_name": ...,
		"definition": ...,	
	}
]
"""

def load_biggen_dataset():
	data_path = "../data/biggen_sample_evals.json"
	with open(data_path, 'r') as file:
		data = json.load(file)

	task2set = {}
	for key, instance in data.items():
		task = instance["task"]
		capability = instance["capability"]
		statement = instance["score_rubric"]["score5_description"]
		if task not in task2set:
			task2set[task] = [instance]
		else:
			task2set[task].append(instance)

	return task2set

def extract_principles_from_biggen():
	task2set = load_biggen_dataset()
	with_gpt_responses = []
	for task, instances in tqdm(task2set.items(), total = len(task2set)):
		main_prompt = "You are given a pair of input and output and a sentence used to evaluate a language model's {} capability.\n".format(instances[0]["capability"])
		
		
		examples = ""
		for idx, instance in enumerate(instances):
			new_idx = idx + 1
			the_idx = "{}\n##Input: {}\n##Output: {}\n##Evaluation sentence: {}\n\n".format(new_idx, instance["input"], instance["reference_answer"], instance["score_rubric"]["score5_description"])
			examples = the_idx + your_task
			full_prompt = main_prompt + examples + "\n\n##Principles:\n"			
			answer = chatgpt_prompter(full_prompt)
			
			instance["gpt_answer"] = answer
			with_gpt_responses.append(instance)

	write_json(with_gpt_responses, "extracted_principles.json")


def main():
	extract_principles_from_biggen()

main()