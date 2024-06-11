# Fig Vishton - June 2024
# This file takes formatted, generated output from GPT and turns it into .pkl files 
# You can run as follows via terminal
#   python3 toPkl.py <filename.txt> <output.pkl>
#   python3toPkl.py Species_exemplars.txt pickle.pkl
# or you can import the functions

import pandas as pd
import pickle 
import re
import sys

#########################################################
#                       FUNCTIONS                       #
#########################################################

def textToList(text):
    """
    textToList: 
        This function takes text that is in a list format 
        and turns it into an actual list in python
        input:  a string
        output: a list
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

        sample = {
            "sample": sampleNum,
            "tokens": tokens,
            "labels": labels
        }
        
        # add to samples list
        samples.append(sample)

    # become dataframe
    samplesDF = pd.DataFrame(samples)

    print(samplesDF)
    return samplesDF


def pickling(dataframe1, outputPath = "default.pkl"):
    """
    pickling: 
        This function takes a dataframe and makes it a pickle file 
        input:  dataframe
        output: creates a pkl file 
    """
    dataframe1.to_pickle(outputPath)

#####################################################
#                       MAIN                        #
#####################################################

sys.argv.pop(0)

fileName = sys.argv.pop(0)

dataframe1 = getInputFile(fileName)
pickling(dataframe1, "pickled.pkl")