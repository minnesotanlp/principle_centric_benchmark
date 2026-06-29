import os
import pandas as pd
import json
import openai
from dotenv import load_dotenv

load_dotenv()

_api_key = os.environ.get("OPENAI_API_KEY")
if not _api_key:
    raise EnvironmentError(
        "OPENAI_API_KEY environment variable not set. "
        "Copy .env.example to .env and fill in your key."
    )
openai.api_key = _api_key

def chatgpt_prompter(input_prompt): 
    completion = openai.chat.completions.create(    
        model = os.environ.get("OPENAI_MODEL", "gpt-4o"),
        messages=[
            {"role": "user", "content": input_prompt}
        ],
        temperature = 0.5,        
    )
    return completion.choices[0].message.content

def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as input_file:
        one_list = json.load(input_file)
    return one_list

def write_json(output_list, output_file):
    with open(output_file, "w", encoding="utf-8") as writer:
        json.dump(output_list, writer, ensure_ascii=False)

    print("Done writing to: ", output_file)
