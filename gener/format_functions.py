def format_single_exemplar(key, value, last_exemplar=False):
    """
    Formats a single extracted exemplar as a custom string.

    Args:
        key: the sample_id for an exemplar.
        value: A dictionary containing the tokens and labels for an exemplar. 

    Returns:
        formatted_exemplar: The extracted exemplar as a custom formatted string. 
    """
    formatted_exemplar = ""

    formatted_exemplar += "\"" + str(key) + "\": {" 
    formatted_exemplar += "\n\t\t\"tokens\": " + str(value["tokens"]) + "," 
    if last_exemplar:
        formatted_exemplar += "\n\t\t\"labels\": " + str(value["labels"]) + "\n    }\n}\n"
    else:
        formatted_exemplar += "\n\t\t\"labels\": " + str(value["labels"]) + "\n    }, \n\t"

    return formatted_exemplar



def format_exemplars_for_export(extracted_exemplars):
    """
    Formats extracted exemplars so they can be exported to a file. 

    Args:
        extracted_exemplars: A dictionary containing the extracted exemplars. 
            Each exemplar should be a dictionary itself with two lists. 

    Returns:
        formatted_exemplars: The extracted exemplars as a custom formatted string. 
    """
    num_of_exemplars = len(extracted_exemplars)

    formatted_exemplars = "{\n\t"
    for count, (key, value) in enumerate(extracted_exemplars.items()):
        if count == num_of_exemplars-1:
            formatted_exemplars += format_single_exemplar(key, value, True)
        else:
            formatted_exemplars += format_single_exemplar(key, value)

    return formatted_exemplars


def format_list_of_entities(user_selected_entity):
    """
    Formats a list of entity names so they are grammatically correct. 

    Args:
        user_selected_entity: A list of strings each representing an user selected entity to 
            generate synthetic data for.
        
    Returns:
        formatted_user_selected_entity: A custom formatted string that contains all user selected entities. 
    """

    if len(user_selected_entity) > 1:
        formatted_user_selected_entity = ""
        for index, entity in enumerate(user_selected_entity):
            if index == len(user_selected_entity)-1:
                formatted_user_selected_entity += f"and {entity}"
            else:
                formatted_user_selected_entity += f"{entity}, "
    else:
        formatted_user_selected_entity = str(user_selected_entity[0])

    return formatted_user_selected_entity