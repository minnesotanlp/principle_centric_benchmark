import pandas as pd
import json
import openai

openai.api_key = ""

def chatgpt_prompter(input_prompt): 
    completion = openai.chat.completions.create(    
        # model = "gpt-4.1",
        model = "gpt-4o",
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

def get_task2input_output_biggen():
    data_path = "biggen_sample_evals.json"
    with open(data_path, 'r') as file:
        data = json.load(file)

    print(len(data.keys()))
    instance = data["planning_travel_plan_0"]
    unique_capabilities = []
    unique_tasks = []

    idx = 1
    task2examples = {}
    for key, instance in data.items():
        task = instance["task"]     
        the_input = instance["input"]
        the_output = instance["reference_answer"]
        pair = {
            "input": the_input,
            "output": the_output
        }
        if task not in task2examples:
            task2examples[task] = [pair]
        else:
            task2examples[task].append(pair)
        
    return task2examples

def get_task2definition_from_biggen():
    task2definition = {
        #CAPABILITY:  planning
        "travel_plan": "Write a travel plan to different destinations with different requirements from the user.",
        "personal_assistant": "You have an agent that is aware of your schedule and priorities. The agent is responsible for scheduling a daily plan of your day!",
        "world_modeling": "Predict the next state of the environment after performing a certain action.",
        "reward_modeling": "Generate a reward function that could assess the actions performed by an agent in a given environment.",
        "compositional_planning": "Construct multiple low-level plans to construct a high-level plan in a modular fashion",
        "constrained_planning": "This task includes a certain intermediate step within the whole planning process", 
        "executable_planning": "Generate an executable plan in an environment that doesn't accept open-ended answers",
        ##CAPABILITY:  theory_of_mind
        "thinking_for_doing": "Task that require to infer what action the opponent would take on the next step. This requires to first infer what the opponent would be thinking about given the possible observations.",
        "guess_the_emotion": "Tasks designed to assess the model's ability to infer emotion from textual cues in a given scenario. This involves understanding subtle language nuances and context to accurately predict the emotional state of characters in a conversation or a narrative.",
        "interplanetary_diplomacy": "Tasks that pre-define a characteristics of a alien species and require the AI model to interpret and convey intentions and desires of why they visited our planet. Given a long conversation, the objective is to generate a summary of the guess or provide additional analysis.",
        "checklist_generation": "Tasks that require to generate a checklist of what each participant of a dialogue is aware of after the conversation. This could include multiple sessions of conversation among various participants.",
        "time_traveler_dilemma": "Assuming a scenario that a time traveler changes a key event in history. The output predicts how various historical figures might react to this change, based on their known beliefs and intentions. For example, how would figures like Gutenberg, Leonardo da Vinci, and Martin Luther react and adapt if the printing press was never invented?",
        "multistep_tom": "Tasks that require to do multi-step reasoning to infer the mental state of the opponent",
        "response_generation": "Tasks that require to generate a next response after comprehending and inferring the mental state of the opponent.",
        "knowledge_graph": "Tasks designed to assess the model's ability to construct a knowledge graph representing both first-order and second-order Theory of Mind.",
        "faux_pas_explanation": "Tasks that require to summarize the conversation and write an explanation given a scenario encompassing a faux-pas emotion. Faux pas is the 'uhoh!' emotion most people would feel when they reveal the reality of the context.",
        "writing_a_speech": "A task where the model is required to write a speech based on provided information about the purpose of the speech and the characteristics of the audience. This task assesses the model's ability to tailor content and language style to fit specific audience demographics and the intended message of the speech.",
        ##CAPABILITY:  instruction_following
        "multi_task_inference": "Tasks require to solve an instruction that is composed of multi-steps at once (e.g., translate-and-summarize)",
        "education_content_creation": "Tasks that require to create new education content such as text books, problem sets, cirriculums, and more! Potentially this data could be used for pre-training on synthetic data.",
        "lexical_constraint": "Tasks that require to follow a given lexical constraint such as word count or starting/ending/including certain words",
        "faithful_explanation": "Language models often excel at generating a list of items, but when we take a closer look at the content, it includes unfactual information. This task requires to explain a list of item accurately, not only looking good at the surface level.",
        "alignment": "Tasks that examine whether the model could adapt to user-defined values by providing in-context demonstrations",
        "executable_actions": "Tasks that require to brainstorm new ideas that are actually actionable, not too abstract",
        "instruction_data_creation": "Tasks that require to create new instruction data using Self-Instruct (i.e., Few-shot Prompting)",
        "false_presupposition": "Tasks that provide an instruction with a false presupposition inside. The crucial part is to not mention about the validity of the false piece of information within the instruction.",
        "semantic_constraint": "Tasks that require to generate an output based on the requested style",
        "ambiguous": "Tasks that provide an instruction that is ambiguous by nature",
        ##CAPABILITY:  reasoning
        "deductive": "Tasks that require to perform deductive reasoning",
        "competition_mwp": "Tasks that require to solve secondary level competition math word problems",
        "abductive": "Tasks that require to perform abductive reasoning",
        "inductive": "Tasks that require to perform inductive reasoning",
        "hypothesis_proposal": "Tasks that require to generate a valid yet intriguing scientific hypothesis",
        "high_school_mwp": "Tasks that require to solve secondary level math word problems",
        "first_order_logic": "Tasks that require to utilize first-order-logic to reason",
        "legal_reason": "Tasks that require to write a consistent and coherent legal statement",
        "table_reason": "Tasks that require to reason over various kinds of tables",
        "math_proof": "Tasks that require to write proofs of secondary level math theorems",
        ##CAPABILITY:  tool_usage
        "multi_step": "Tasks requiring the breakdown of a given task into subtasks and using tools appropriately",
        "web_browsing": "Tasks that require generating actionable items while browsing through the web",
        "coding_for_math": "Tasks that require to use a coding terminal to solve a math word problem",
        "item_recommendation": "Tasks that require recommending a requested item using multiple APIs and search engines",
        "tool_making": "Tasks that require creating new set of tools for problem-solving",
        "api_documentation": "Tasks that require reading through a set of documentations to write a piece of code that uses multiple APIs",
        "search_engine": "Tasks that involve the use of search engines and generating effective search queries",
        ##CAPABILITY:  grounding
        "temporal_grounding": "Tasks that require to ground on a given time when additional context is given",
        "false_context": "Tasks that require to generate an output when false information is given as additional context.",
        "system_prompt_vs_demo": "Tasks where the system prompt and demonstrations are somehow related but show different patterns.",
        "role_playing": "Tasks that require to simulate a role of a certain historical figure or character",
        "demo_vs_instruction": "Tasks that where the instruction and demonstrations are somehow related but show different patterns",
        "follow_the_system_prompt": "Tasks that require to follow an unusual and challenging system prompt that requires acting beyond as a 'Helpful Assistant'",
        "json_csv_xml": "Tasks that require to process a given complicated data (at least 100 rows) in the requested output format (e.g., nested json format)",
        "multi_source": "Tasks that require to generate based on multiple knowledge sources that might potentially conflict with one another",
        "simulator": "Tasks that require the LLM to function as a simulator of the requested target (e.g., user or a terminal system)",
        "system_prompt_vs_instruction": "Tasks where the system prompt and instruction are somehow related but show different patterns",
        ##CAPABILITY:  refinement
        "rationale_revision": "Tasks that require edit the rationale used to derive an answer. The wrong rationale and feedback from the user pinpointing what is wrong is given as additional context",
        "replanning": "Tasks that require to regenerate a plan given certain observations from the environment",
        "llm_judge_absolute": "Tasks that require to write a language feedback and a corresponding score between the range of 1 to 5",
        "code_revision": "Tasks that require to fix a wrong piece of code. A short comment of what is wrong and the terminal log is given as additional context",
        "revision_with_tools": "Tasks that require to revise the output given a set of tools such as a search engine or a set of APIs. A description of how to use the tools are given as input",
        "self_correction": "Tasks that require to improve a given text. No additional feedback is given",
        "essay_revision": "Tasks that require to revise a 3-5 paragraph essay using feedbacks from a user",
        "llm_judge_relative": "Tasks that require to compare two different responses in order to write a language feedback and a decide which is a better response",
        ##CAPABILITY:  safety
        "knowledge_unlearning": "Tasks that require to unlearn certain concepts related to privacy issues by providing a set of in-context demonstrations",
        "if_else_statements": "Tasks that require to generate a piece of code that divides priority in a given situation",
        "mentioning_potential_harm": "Tasks that require explaining a list of items. The main objective is to check whether at least one of the items mentions about the potential harm when executing an action",
        "safety_alignment": "Tasks that red team language models to generate toxic content by providing a set of in-context demonstrations",
        "honesty": "Tasks that provide synthetic information as additional input and probes the model to state whether it knows or does not know a queried information.",
        "keep_a_secret": "Tasks that prompt language models to generate certain informations. The model should not generate sensitive information within its responses",
        "determine_what_is_wrong": "Tasks that require to to explain what is controversial about a given text. The model should be able to state historical backgrounds, provenance, and examples of why it is toxic to use such term",
        "moral_belief": "Tasks necessitates possessing an opinion in a moral dilemma situation",
    }
    return task2definition

#this function is loading extracted principles
def load_extracted_principle_and_definition():
    extracted_principles_data = pd.read_csv("extracted_principles.tsv", sep="\t")
    temp_dict = {}
    for idx, row in extracted_principles_data.iterrows():        
        task_name = row["task_name"]
        principle = row["final_principle_name"]
        principle_definition = row["final_definition"]
        if task_name not in temp_dict:
            temp_dict[task_name] = {principle: principle_definition}
        else:
            temp_dict[task_name][principle] = principle_definition
    
    return temp_dict

#this function is loading generated principles
def load_zeroshot_principle_and_definition():
    extracted_principles_data = pd.read_csv("generated_principles.tsv", sep="\t")
    temp_dict = {}
    for idx, row in extracted_principles_data.iterrows():        
        task_name = row["task"]
        principle = row["principle_name"]
        principle_definition = row["principle_definition"]
        if task_name not in temp_dict:
            temp_dict[task_name] = {principle: principle_definition}
        else:
            temp_dict[task_name][principle] = principle_definition
    
    return temp_dict