def get_label_counts(data_df, label_column='Label'):
    """
    Counts occurrences of all labels in a DataFrame column where each cell contains a list of labels,
    assuming there are no NaN values in the column.
    
    Args:
        data_df: pandas.DataFrame containing the data.
        label_column: str, the name of the column containing the lists of labels.
    
    Returns:
        label_counts: A dictionary with labels as keys and their counts as values.
    """
    label_counts = {}
    for labels in data_df[label_column]:
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
    return label_counts



def add_sentence_tag(dataframe, label_frequency, user_selected_entites, extract_exemplars):
    """
    Updates 'SentenceTag' column in the DataFrame with the label that has the lowest frequency
    according to the provided label_frequency dictionary.
    
    Args:
        dataframe: pd.DataFrame with a 'Label' column containing lists of labels.
        label_frequency: Dictionary with labels as keys and their frequencies as values.
        user_selected_entites: Filter the labels to a smaller set we wish to consider.
        extract_exemplars: True or flase flag for whether we are extracting exemplars or preparing for continual learning
    """
    
    # Add an empty 'SentenceTag' column to the DataFrame if it doesn't exist
    if 'SentenceTag' not in dataframe.columns:
        dataframe['SentenceTag'] = None

    # Function to find the label with the lowest frequency
    def find_lowest_frequency_label(label_list, user_selected_entites, extract_exemplars):
        # Filter out labels that are not in the label_frequency dictionary
        filtered_labels = [label for label in label_list if label in label_frequency]

        if not filtered_labels: # If filtered_labels is empty, return None
            print("Label Mismatch!")
            return None
        
        if extract_exemplars:
            unique_labels = set(filtered_labels) # Get the unique labels present in the label list
            if len(unique_labels) == 2:
                # Return the non-O label 
                if "O" in unique_labels:
                    unique_labels.remove("O")
                    return list(unique_labels)[0]
                else:
                    return list(unique_labels)[0]
            else:
                return "O"

        # Filter down to only the labels we wish to partition the continual learning buckets with.
        user_filtered_labels = [label for label in filtered_labels if label in user_selected_entites]

        if not user_filtered_labels:
            # print("No user selected entities present in the sentence. Returning the \"O\" class.")
            # print(filtered_labels)
            user_filtered_labels = ["O"]
        
        # Find the label with the lowest frequency
        return min(user_filtered_labels, key=lambda label: label_frequency[label])

    # Apply the function to each row and update the 'SentenceTag' column
    for index, row in dataframe.iterrows():
        lowest_freq_label = find_lowest_frequency_label(row['Label'], user_selected_entites, extract_exemplars)
        dataframe.at[index, 'SentenceTag'] = lowest_freq_label
