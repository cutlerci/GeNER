import logging
from gener.query_openai_model import query_openai_model

def query_llm(generative_model, generative_batch_size, generate_num_samples):
    """
    Queries a generative model based on the user selection.

    Args:
        generative_model: The generative model to use when making queries.
        generative_batch_size: The number of samples to generate in a single conversation.
        generate_num_samples: The total number of samples to generate using each prompt.
    """
    
    logging.info(f"Querying {generative_model}")

    if generative_model == "gpt-3.5-turbo-0125" or generative_model == "gpt-4-0125-preview":
        query_openai_model(generative_model, generative_batch_size, generate_num_samples)
