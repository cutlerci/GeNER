import os
import logging
from openai import OpenAI

def query_openai_model(generative_model, generative_batch_size, generate_num_samples):
    """
    Queries an OpenAI based generative model using available prompts.

    Args:
        generative_model: The OpenAI generative model to use when making queries.
        generative_batch_size: The number of samples to generate in a single conversation.
        generate_num_samples: The total number of samples to generate using each prompt.
    """

    logging.info(f"Reading available prompts.")
    prompts = {}

    # If ad-hoc prompts exist, read them into memory.
    if os.path.exists("synthetic_data/prompts/ad_hoc"):
        for prompt_file in os.listdir("synthetic_data/prompts/ad_hoc"):
            prompt_path = os.path.join("synthetic_data/prompts/ad_hoc", prompt_file)
            with open(prompt_path, 'r') as file:
                prompt = file.read()
            prompts[f"ad_hoc/{prompt_file[:-4]}"] = prompt
            os.makedirs("synthetic_data/outputs/ad_hoc", exist_ok=True)

    # If kee prompts exist, read them into memory.
    if os.path.exists("synthetic_data/prompts/kee"):
        for prompt_file in os.listdir("synthetic_data/prompts/kee"):
            prompt_path = os.path.join("synthetic_data/prompts/kee", prompt_file)
            with open(prompt_path, 'r') as file:
                prompt = file.read()
            prompts[f"kee/{prompt_file[:-4]}"] = prompt
            os.makedirs("synthetic_data/outputs/kee", exist_ok=True)

    number_of_batches = int(generate_num_samples / generative_batch_size)

    for prompt_name, prompt in prompts.items():
        for generative_batch in range(0, number_of_batches):
            logging.info(f"Generating {prompt_name}, batch {generative_batch}.")
            client = OpenAI()
            my_messages=[{"role": "user", "content": prompt}]

            output_file_path = f"synthetic_data/outputs/{prompt_name}_batch_{generative_batch}.txt"

            for query_count in range(0, int(generative_batch_size/5)):   
                completion = client.chat.completions.create(
                model=generative_model,
                messages=my_messages
                ) 

                with open(output_file_path, 'a' if os.path.exists(output_file_path) else 'w') as file:
                    file.write(completion.choices[0].message.content)

                my_messages += [{"role": "assistant", "content": completion.choices[0].message.content}]
                my_messages += [{"role": "user", "content": "Create 5 more."}]
    logging.info(f"LLM data generation complete.\n\n")
