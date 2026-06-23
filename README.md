# principle_centric_benchmark
Dataset and code for the ACL 2026 EvalEval Paper paper ["From Rubrics to Recipe: Principle-Centric Benchmark for Evaluating Large Language Models"](https://openreview.net/pdf?id=yQhhR9cKE1). 

## Dataset
GPT-data with principles
* data/gpt_data_with_extracted_principles
* data/gpt_data_with_generated_principles

Other LLMs' outputs:
* data/gemma_outputs
* data/qwen3b_outputs
* data/qwen7b_outputs
* data/qwen14b_outputs


## Code
* `principle_extractor.py`: extract principles from BigGen rubric 
* `principle_generator.py`: generate principles based on BigGen data
* `data_generator.py`: generate input-output pairs based on principles
* `llm_as_judge.py`: evaluate open LLMs' outputs with principles as rubric
* `utils.py`

## Citation
```
@inproceedings{hayati-etal-2026-from,
    title = "From Rubrics to Recipe: Principle-Centric Benchmark for Evaluating Large Language Models",
    author = "Hayati, Shirley Anugrah and Wang, Ruizi and Kang, Dongyeop",
    booktitle = "ACL 2026 Workshop on Evaluating Evaluations (EvalEval)",
    month = jul,
    year = "2026",
    address = "San Diego, United States of America",
    publisher = "Association for Computational Linguistics",
}
```
