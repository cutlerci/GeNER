import os
import re
from gener.format_functions import format_list_of_entities, format_single_exemplar


def build_ad_hoc_prompts(formatted_entities, exemplars, partitioned_exemplar_keys, num_prompts, num_shots):
    """
    Construct prompts using the ad-hoc prompting strategy.

    Args:
        formatted_entities: A custom formatted string listing the user selected entities.
        exemplars: A dictionary of data instances extracted as exemplars.
        partitioned_exemplar_keys: A list of lists partitioning the exemplars into groups. 
            One group of exemplars for each prompt.
        num_prompts: The number of prompts to be created. Used for all requested prompting strategies.
        num_shots: The number of examples to be included in each prompt.
    """

    # <><><> Read in the ad-hoc prompt strategy components <><><>
    with open('gener/prompt_components/ad_hoc_upper.txt', 'r') as file:
        upper_portion = file.read()

    with open('gener/prompt_components/ad_hoc_lower.txt', 'r') as file:
        lower_portion = file.read()
    # Dynamically update the prompt component for the user selected entities
    lower_portion = re.sub(r"<ENTITY>", formatted_entities, lower_portion)


    # <><><> Construct ad-hoc prompts <><><>
    for prompt_id in range(0, num_prompts):
        prompt = ""
        prompt += upper_portion
        prompt += "\n\n\nHere are five examples:\n\n{\n\t"

        # Insert the exemplars.
        for sample_id, key in enumerate(partitioned_exemplar_keys[prompt_id]):
            if sample_id == num_shots-1:
                prompt += format_single_exemplar(f"sample{sample_id}", exemplars[key], last_exemplar=True)
            else:
                prompt += format_single_exemplar(f"sample{sample_id}", exemplars[key], last_exemplar=False)
        
        prompt += "\n"
        prompt += lower_portion

        # Save the prompt to a file.
        output_file_path = f"synthetic_data/prompts/ad_hoc/prompt_{prompt_id}.txt"
        with open(output_file_path, 'w') as file:
            file.write(prompt)



def build_kee_prompts(user_selected_entity, formatted_entities, exemplars, partitioned_exemplar_keys, num_prompts, num_shots):
    """
    Construct prompts using the kee prompting strategy.

    Args:
        user_selected_entity: A list of entities to generate synthetic data for.
        formatted_entities: A custom formatted string listing the user selected entities.
        exemplars: A dictionary of data instances extracted as exemplars.
        partitioned_exemplar_keys: A list of lists partitioning the exemplars into groups. 
            One group of exemplars for each prompt.
        num_prompts: The number of prompts to be created. Used for all requested prompting strategies.
        num_shots: The number of examples to be included in each prompt.
    """

    # <><><> Read in the kee prompt strategy components <><><>
    with open('gener/prompt_components/task_description.txt', 'r') as file:
        task_description = file.read()
    
    entity_definitions = "<Entity Definitions>\n"
    for entity in user_selected_entity:
        with open(f'gener/prompt_components/entity_definitions/{entity.lower()}_definition.txt', 'r') as file:
            entity_definitions += file.read()
            entity_definitions += "\n\n"

    with open('gener/prompt_components/task_emphasis.txt', 'r') as file:
        task_emphasis = file.read()

    with open('gener/prompt_components/prompt.txt', 'r') as file:
        prompt_component = file.read()
    # Dynamically update the prompt component for the user selected entities
    prompt_component = re.sub(r"<ENTITY>", formatted_entities, prompt_component)


    # <><><> Construct kee prompts <><><>
    for prompt_id in range(0, num_prompts):
        prompt = ""
        prompt += task_description + "\n\n" + entity_definitions + task_emphasis
        prompt += "\n\n<Task Examples>\nHere are five examples:\n\n{\n\t"

        # Insert the exemplars.
        for sample_id, key in enumerate(partitioned_exemplar_keys[prompt_id]):
            if sample_id == num_shots-1:
                prompt += format_single_exemplar(f"sample{sample_id}", exemplars[key], last_exemplar=True)
            else:
                prompt += format_single_exemplar(f"sample{sample_id}", exemplars[key], last_exemplar=False)
        
        prompt += "\n"
        prompt += prompt_component

        # Save the prompt to a file.
        output_file_path = f"synthetic_data/prompts/kee/prompt_{prompt_id}.txt"
        with open(output_file_path, 'w') as file:
            file.write(prompt)



def build_prompts(user_selected_entity, exemplars, num_prompts, num_shots, prompt_strategies):
    """
    Construct the prompts used to query the generative large language model.

    Args:
        user_selected_entity: A list of entities to generate synthetic data for.
        exemplars: A dictionary of data instances extracted as exemplars.
        num_prompts: The number of prompts to be created. Used for all requested prompting strategies.
        num_shots: The number of examples to be included in each prompt.
        prompt_strategies: The prompting strategy to use when creating the prompts.
    """

    formatted_entities = format_list_of_entities(user_selected_entity)

    exemplar_keys = list(exemplars.keys())
    partitioned_exemplar_keys = [exemplar_keys[i:i+num_shots] for i in range(0, len(exemplar_keys), num_shots)]

    if prompt_strategies == "all":
        os.makedirs("synthetic_data/prompts/ad_hoc/", exist_ok=True)
        os.makedirs("synthetic_data/prompts/kee/", exist_ok=True)

        build_ad_hoc_prompts(formatted_entities, exemplars, partitioned_exemplar_keys, num_prompts, num_shots)
        build_kee_prompts(user_selected_entity, formatted_entities, exemplars, partitioned_exemplar_keys, num_prompts, num_shots)

    elif prompt_strategies == "ad-hoc":
        os.makedirs("synthetic_data/prompts/ad_hoc/", exist_ok=True)

        build_ad_hoc_prompts(formatted_entities, exemplars, partitioned_exemplar_keys, num_prompts, num_shots)

    elif prompt_strategies == "kee":
        os.makedirs("synthetic_data/prompts/kee/", exist_ok=True)

        build_kee_prompts(user_selected_entity, formatted_entities, exemplars, partitioned_exemplar_keys, num_prompts, num_shots)
   