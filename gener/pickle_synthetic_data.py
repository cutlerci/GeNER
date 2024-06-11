import os
import re
import logging
import pandas as pd



def textToList(text):
    """
    textToList: 
        This function takes text that is in a list format 
        and turns it into an actual list in python
        input:  a string
        output: a list
    
    Fig Vishton - June 2024
    """
    # split by the inevitable comma and space 
    tempList = text.split(", ")
    outputList = []

    # take off surrounding single or double quotes, append to output list
    for item in tempList:
        item2 = item [1:-1]
        outputList.append(item2)

    return outputList



def getInputFile(fileName):
    """
    getInputFile: 
        This function gets input from a file to be made into a pkl file 
        (other functions could be defined to get input from somewhere different)
        input:  a filename/filepath (string)
        output: list of dicts

    Fig Vishton - June 2024
    """
    # open file 
    with open(fileName, 'r') as file:
        text = file.read()

    # get the samples (in correct format)
    pattern = r'("sample\d+"): {\s*("tokens":) \[(.*?)\],\s*("labels":) \[(.*?)\]\s*}'

    matches = re.findall(pattern, text)
    samples = []

    # turn into dicts 
    for match in matches:

        sampleNum = match[0].replace(f"\"", "")
        tokens = textToList(match[2])
        labels = textToList(match[4])

        # If the generated labels list is shorter than the generated tokens, extend the label list with 'O'
        if len(labels) < len(tokens):
            labels.extend(['O'] * (len(tokens) - len(labels)))  

        # If the generated labels list is longer than the generated tokens, remove the labels from the end
        if len(labels) > len(tokens):
            labels = labels[:len(tokens)]
    
        sample = {
            "sample": sampleNum,
            "tokens": tokens,
            "labels": labels
        }
        
        # add to samples list
        samples.append(sample)

    # become dataframe
    samplesDF = pd.DataFrame(samples)

    return samplesDF



def pickling(dataframe1, outputPath = "synthetic_data.pkl"):
    """
    pickling: 
        This function takes a dataframe and makes it a pickle file 
        input:  dataframe
        output: creates a pkl file 
    
    Fig Vishton - June 2024
    """
    dataframe1.to_pickle(outputPath)



def pickle_ad_hoc_synthetic_data(generative_batch_size):
    """
    Reformats and pickles the generated outputs from the ad_hoc style outputs into a Pandas dataframe.

    Args:
        generative_batch_size: The number of samples to generate in a single conversation.
    """

    ad_hoc_dataframe = pd.DataFrame()

    for output_file in os.listdir("synthetic_data/outputs/ad_hoc"):
        output_path = os.path.join("synthetic_data/outputs/ad_hoc", output_file)

        # Read generated samples into a Pandas Dataframe
        dataframe = getInputFile(output_path)

        num_extracted_synthetic_samples = len(dataframe)

        # Check for extracting mismatch. Indicates a generated sample that was not in the correct format. 
        if num_extracted_synthetic_samples < generative_batch_size:
            logging.warning(f"Synthetic data mismatch in file {output_path}. Expected to save {generative_batch_size} synthetic samples, actually saved {num_extracted_synthetic_samples}.\n")
    
        ad_hoc_dataframe = pd.concat([ad_hoc_dataframe, dataframe], ignore_index=True)

    ad_hoc_dataframe = ad_hoc_dataframe.drop(columns=['sample'])

    pickling(ad_hoc_dataframe, "synthetic_data/synthetic_data_ad_hoc.pkl")



def pickle_kee_synthetic_data(generative_batch_size):
    """
    Reformats and pickles the generated outputs from the kee style outputs into a Pandas dataframe.

    Args:
        generative_batch_size: The number of samples to generate in a single conversation.
    """
    kee_dataframe = pd.DataFrame()

    for output_file in os.listdir("synthetic_data/outputs/kee"):
        output_path = os.path.join("synthetic_data/outputs/kee", output_file)

        # Read generated samples into a Pandas Dataframe
        dataframe = getInputFile(output_path)

        num_extracted_synthetic_samples = len(dataframe)

        # Check for extracting mismatch. Indicates a generated sample that was not in the correct format. 
        if num_extracted_synthetic_samples < generative_batch_size:
            logging.warning(f"Synthetic data mismatch in file {output_path}. Expected to save {generative_batch_size} synthetic samples, actually saved {num_extracted_synthetic_samples}.\n")
    
        kee_dataframe = pd.concat([kee_dataframe, dataframe], ignore_index=True)

    kee_dataframe = kee_dataframe.drop(columns=['sample'])

    pickling(kee_dataframe, "synthetic_data/synthetic_data_kee.pkl")



def pickle_synthetic_data(prompt_strategies, generative_batch_size):
    """
    Reformat and pickle the generated outputs from the large language model.

    Args:
        prompt_strategies: The prompting strategy to use when creating the prompts.
        generative_batch_size: The number of samples to generate in a single conversation.
    """

    if prompt_strategies == "all":
        pickle_ad_hoc_synthetic_data(generative_batch_size)
        pickle_kee_synthetic_data(generative_batch_size)

    elif prompt_strategies == "ad-hoc":
        pickle_ad_hoc_synthetic_data(generative_batch_size)

    elif prompt_strategies == "kee":
        pickle_kee_synthetic_data(generative_batch_size)
