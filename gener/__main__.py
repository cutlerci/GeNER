import argparse
import logging
import datetime as dt
from sys import argv

from gener.extract_exemplars import extract_exemplars
from gener.build_prompts import build_prompts
from gener.query_llm import query_llm
from gener.pickle_synthetic_data import pickle_synthetic_data

def create_parser():
    """
    Sets up the command line argument parser.

    Returns: 
        parser: The argparse parser object created.
    """
    parser = argparse.ArgumentParser(prog='gener', description="Create synthetic NER data using a generative large language model.")
    
    # Required Arguments
    parser.add_argument('dataset_path', help='The file path to the preprocessed authentic dataset.')
    parser.add_argument('entities', nargs='+',
                        help='Specify the entities for which synthetic data should be generated.')

    # Optional Arguments
    parser.add_argument('-np', '--num_prompts', default=10, type=int,
                        help='Specify the number of prompts to be created. Used for all requested prompting strategies. (default: %(default)s)')
    parser.add_argument('-ns', '--num_shots', default=5, type=int,
                        help='Specify the number of shots to be included in each prompt. (default: %(default)s)')
    parser.add_argument('-ps', '--prompt_strategies', default='all', choices=['all', 'ad-hoc', 'kee'],
                        help='Specify the prompting strategy to use. (default: %(default)s)')
    parser.add_argument('-llm', '--generative_model', default='gpt-4-0125-preview', choices=['gpt-3.5-turbo-0125','gpt-4-0125-preview'],
                        help='Specify the generative large langauge model to use. (default: %(default)s)')
    parser.add_argument('-gbs', '--generative_batch_size', default=50, type=int,
                        help='Specify the number of samples to generate up to in a single query. (default: %(default)s)')
    parser.add_argument('-gns', '--generate_num_samples', default=100, type=int,
                        help='Specify the number of samples to generate in total for a single prompt. (default: %(default)s)')
    return parser



def add_calculated_args(args):
    """
    Adds any additional args derived from the user provided arguments.

    Args:
        args: Populated namespace from parsing user provided arguments.

    Returns: 
        args: Populated namespace containing addtional arguments that are derived from the user provided arguments.
    """
   
    # Adjust the number of requested exemplars based on the number of user selected entities.
    args.requested_exemplars = int(args.num_prompts * (args.num_shots / len(args.entities)))

    return args


def validate_args(args):
    """
    Custom validation of the user provided arguments.

    Args:
        args: Populated namespace from parsing user provided arguments.
    """

    if args.num_prompts <= 0:
        raise ValueError("Num_prompts must be greater than zero.")
    
    if args.num_shots <= 0:
        raise ValueError("num_shots must be greater than zero.")
    
    if args.num_shots % len(args.entities) != 0:
        raise ValueError("Num_shots must be evenly divisible by the number of entities for which synthetic data should be generated.")

    if args.generate_num_samples % args.generative_batch_size != 0:
        raise ValueError("generate_num_samples must be evenly divisible by generative_batch_size.")


# In order of priority
# TODO: Unit testing
# TODO: Add functionality to override the number of requested prompts if not enough suitable instances exist.

def main():
    # Argument Parser Setup
    parser = create_parser()
    args = parser.parse_args()
    validate_args(args)
    args = add_calculated_args(args)

    # Logging Setup
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='gener.log', level=logging.INFO)
    logging.info(f"GeNER \nCommand: python -m gener {' '.join(argv[1:])}")
    start_time = dt.datetime.now()
    start_timestamp = start_time.strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f'\n\nSTART TIME: {start_timestamp}\n')

    # Synthetic Data Generation
    extracted_exemplars = extract_exemplars(preprocessed_data_path=args.dataset_path, 
                                            user_selected_entity=args.entities, 
                                            num_exemplars=args.requested_exemplars)
    
    build_prompts(user_selected_entity=args.entities,
                  exemplars=extracted_exemplars,
                  num_prompts=args.num_prompts,
                  num_shots=args.num_shots,
                  prompt_strategies=args.prompt_strategies)
    
    query_llm(generative_model=args.generative_model,
              generative_batch_size=args.generative_batch_size,
              generate_num_samples=args.generate_num_samples)

    pickle_synthetic_data(prompt_strategies=args.prompt_strategies,
                          generative_batch_size=args.generative_batch_size)

    # Calculate/print end time
    end_time = dt.datetime.now()
    end_timestamp = end_time.strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f'END TIME: {end_timestamp}')

    # Calculate/print time elapsed
    elapsed_time: dt.timedelta = end_time - start_time
    logging.info(f'TIME ELAPSED: {elapsed_time}')


if __name__ == "__main__":
    main()
