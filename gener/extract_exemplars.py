import os
import random
import logging
import pandas as pd

from gener.data_preprocessing import get_label_counts, add_sentence_tag
from gener.format_functions import format_exemplars_for_export



def locate_suitable_exemplars(preprocessed_data_df, user_selected_entity):
    """
    Locate indices of data instances identified as suitable for the user selected entity types.

    Args:
        preprocessed_data_df: The dataframe in which suitable instances should be located.
        user_selected_entity: The user selected entities for which to extract exemplars.

    Returns:
        entity_count: Number of suitable exemplars.
        exemplar_indices: List of indices corressponding suitable data instances.
    """
    entity_count = 0
    exemplar_indices = []
    for index, row in preprocessed_data_df.iterrows():
        if row["SentenceTag"] == user_selected_entity:
            entity_count += 1
            exemplar_indices.append(index)

    return entity_count, exemplar_indices



def extract_exemplars(preprocessed_data_path, user_selected_entity, num_exemplars):
    """
    Extract authentic data instances to serve exemplars. 

    For each user selected entity, extracts the requested number of authentic data instances.
        Extracted instances must be suitable and will serve as shots within the prompts.
        They are saved to a file in a custom format as well as returned as a Python dictionary 
        for subsequent steps in the synthetic data generation process.

    Args:
        preprocessed_data_path: The file path to the preprocssed authentic dataset.
            The dataset should be a pickled pandas dataframe with the following two columns. 
                "Label": A column containing lists of labels. 
                "Word": A column containing lists of tokens.
        user_selected_entity: A list of entities to extract exemplars for.
        num_exemplars: The number of exemplars to extract for each entity. 

    Returns:
        extracted_exemplars: A dictionary of data instances extracted as exemplars.
    """

    # <><><> Further refine preprocessed data <><><>
    preprocessed_data_df = pd.read_pickle(preprocessed_data_path) 
    
    # Get the class label frequency counts.
    count_dict = get_label_counts(preprocessed_data_df) 

    # Identify the least frequently seen entity in each instance.
    add_sentence_tag(preprocessed_data_df, count_dict)
    preprocessed_data_df = preprocessed_data_df.reset_index()


    # <><><> Extract Exemplars <><><>
    # Identify instances that are suitable to serve as exemplars for each user selected entity type.
    logging.info(f"Identifying instances that are suitable to serve as {user_selected_entity[0]} exemplars.")
    entity_count, exemplar_indices = locate_suitable_exemplars(preprocessed_data_df, user_selected_entity[0])


    # Verify enough suitable instances exist.
    if not entity_count: 
        raise ValueError(f"No suitable exemplars exist. Not enough to select the requested {num_exemplars}.")

    if entity_count < num_exemplars: 
        raise ValueError(f"{entity_count} suitable exemplars exist. Not enough to select the requested {num_exemplars}. Try lowering the number of requested prompts.")

    # Randomly select num_exemplars from identified suitable instances.
    random_numbers = random.sample(range(0, entity_count), num_exemplars)

    # Extract exemplars by building a dictionary of the randomly selected instances.
    logging.info(f"Extracting exemplars.")
    extracted_exemplars = {}
    for exemplar_id, random_number in enumerate(random_numbers):
        # Select a random suitable exemplar.
        index_of_random_exemplar = exemplar_indices[random_number]
        random_exemplar = preprocessed_data_df.loc[preprocessed_data_df.index[index_of_random_exemplar], ["Label","Word"]]

        # Save it as a dictionary.
        extracted_exemplars[f"sample{exemplar_id}"] = {"tokens": random_exemplar["Word"],
                                                    "labels": random_exemplar["Label"]}

    # <><><> Save Exemplars to a File <><><>
    os.makedirs("synthetic_data/exemplars/", exist_ok=True)
    output_file = f"synthetic_data/exemplars/{user_selected_entity[0]}_exemplars.txt"
    
    logging.info(f"Formatting exemplars to be saved to a file.")
    formatted_exemplars = format_exemplars_for_export(extracted_exemplars)

    logging.info(f"Saving exemplars to {output_file}\n\n")    
    with open(output_file, 'a' if os.path.exists(output_file) else 'w') as file:
        file.write(formatted_exemplars)
    
    return extracted_exemplars
    