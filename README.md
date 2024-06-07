<img src="gener_logo.jpg" alt="GeNER Logo" align="right" width="300" height="300">

# Generative NER (GeNER): Synthetic NER Data Creation Tool

GeNER is a versatile tool for generating synthetic named entity recognition (NER) training data using large language models (LLMs). It currently supports OpenAI models but has the capacity to be expanded to include a wider selection of LLMs. The tool offers an end-to-end pipeline for generating synthetic NER data, including exemplar extraction, prompt creation, model querying, and data reformatting. Additionally, it supports multiple prompting strategies, with two currently implemented.

# Features
- **Modular LLM Support**: Easily expandable to support various large language models, with initial support for OpenAI models.
- **End-to-End Pipeline**: Comprehensive workflow from exemplar extraction to data reformatting.
- **Multiple Prompting Strategies**: Includes support for various prompting strategies.
- Active community development spearheaded and maintained by [NLP@VCU](https://nlp.cs.vcu.edu/).

## Installation Instructions
GeNER can be installed for general use or development/research purposes. To install GeNER, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/cutlerci/GeNER.git
cd GeNER
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Install GeNER:
```bash
pip install -e .
```

## Usage
GeNER is used through its command line interface. To run the project, execute the following command:
```bash
python3 -m gener DATASET_PATH ENTITY [OPTIONS]
```
Replace DATASET_PATH with the file path to the preprocessed authentic dataset and ENTITY with the entity for which synthetic data should be generated.

### Optional Arguments
* -np, --num_prompts: Specify the number of prompts to be created. Used for all requested prompting strategies. (default: 10)
* -ns, --num_shots: Specify the number of shots to be included in each prompt. (default: 5)
* -ps, --prompt_strategies: Specify the prompting strategy to use. Options: 'all', 'ad-hoc', 'kee' (default: all)
* -llm, --generative_model: Specify the generative large language model to use. Options: 'gpt-3.5-turbo-0125', 'gpt-4-0125-preview' (default: gpt-4-0125-preview)
* -gbs, --generative_batch_size: Specify the number of samples to generate up to in a single query. (default: 50)
* -gns, --generate_num_samples: Specify the number of samples to generate in total for a single prompt. (default: 100)

### Example
For example, to generate synthetic NER data for the entity 'Species' using the file 'preprocessed_data.pkl':

```bash
python3 -m gener preprocessed_data.pkl Species -llm gpt-3.5-turbo-0125 -np 2 -gbs 5 -gns 10
```

## Authors
Current contributors: Charles Cutler, Scott Taylor, Fig Vishton, and Bridget T. McInnes

## Acknowledgments
![alt text](https://nlp.cs.vcu.edu/images/vcu_head_logo "VCU")[VCU Natural Language Processing Lab](https://nlp.cs.vcu.edu/) 

