import pandas as pd
import numpy as np

def load_data(filepath):

    # List of valid extensions
    valid_extensions = ['csv', 'txt', 'tsv']
    if not any(filepath.endswith(ext) for ext in valid_extensions):
        raise ValueError(f"Invalid file extension. Supported formats are: {valid_extensions}")

    # Load data into a pandas DataFrame depending on file extension
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith('.txt'):
        df = pd.read_csv(filepath, sep=',')  # Assuming comma-separated values in .txt
    elif filepath.endswith('.tsv'):
        df = pd.read_csv(filepath, sep='\t') # Assuming tab-separated values in .tsv
   
    return df

def calc_majority_vote(data, attr):

    # Obtain unique values for attribute in dataset
    values = set(data[attr])

    # Raise error if no values
    if len(values) == 0:
        raise ValueError(f"No values provided for attribute {attr}.")

    # Get unique values
    unique_values = set(values)

    # Convert attribute values to a numpy array
    attr_values = np.array(data[attr])

    # Count occurrences of each value
    counts = {val: np.sum(attr_values == val) for val in unique_values}

    # Return the most common value
    return max(counts, key=counts.get)

# Calculate entropy given a probability based on formula -p*log2(p)
def calc_entropy(data, target):

    # Get unique values and their probabilities
    values = set(data[target])
    probs = {val: np.sum(data[target] == val) / len(data) for val in values}

    # Calculate entropy using the formula
    entropy = sum(-p * np.log2(p) for p in probs.values() if p > 0)

    return entropy

def calc_conditional_entropy(data, target, feature):
    
    # Get unique values of the feature
    unique_values = set(data[feature])
    conditional_entropy = 0.0

    # For each value in unique_values...
    for val in unique_values:

        # Grab a subset of dataset where feature value equals val
        subset = data[data[feature] == val]
        assert len(subset) > 0, "Subset should not be empty"

        # Calculate the probability of this feature value (f_v)
        prob = len(subset) / len(data)
 
        conditional_entropy += prob * calc_entropy(subset, target)

    return conditional_entropy

def calc_mutual_information(entropy, cond_entropy):
    return entropy - cond_entropy

def calc_error_rate(data, target, prediction):
    
    # If no data... return 0
    if len(data) == 0:
        return 0.0
    
    # Get number of incorrect predictions by comparing values in target and prediction columns
    incorrect = int((data[target] != data[prediction]).sum())
    
    # Return error rate; We'll use this to compute accuracy % as well
    return incorrect / len(data)

def best_split(data, features, target, threshold=0.0):
    
    # Bookkeeping
    entropy_target = calc_entropy(data, target)
    best_feature, best_value = None, None
    best_left, best_right = None, None
    best_mutual_info = 0.0
    
    # For feature in features...
    for feat in features:

        # Calculate conditional entropy for feature over full data
        cond_entropy = calc_conditional_entropy(data, target, feat)
        
        # Calculate mutual information
        mutual_info = calc_mutual_information(entropy_target, cond_entropy)
        # print(f"\nEntropy H({target}): {entropy_target}")
        # print(f"Conditional Entropy H({target}|{feat}): {cond_entropy}")
        # print(f"Mutual Information I({target}; {feat}): {mutual_info}")
        
        # If current mutual information greater than best & greater than threshold...
        if mutual_info > best_mutual_info and mutual_info > threshold:
            
            # Split the dataset on this feature
            v = calc_majority_vote(data, feat)         # Using this to split on. Its just most occuring value
            left_subset = data[data[feat] == v]
            right_subset = data[data[feat] != v]
            
            # print(left_subset.head())
            # print(right_subset.head())

            # Make sure both are non-empty
            if len(left_subset) == 0 or len(right_subset) == 0:
                continue
        
            # Update bookkeeping
            best_mutual_info = mutual_info
            best_feature = feat
            best_value = v
            
            # Only care about indexes for this
            best_left = left_subset.index.tolist()
            best_right = right_subset.index.tolist()

            
    # If no best_feature...
    if best_feature is None:
        return None
            
    return{
        'feature': best_feature,
        'value': best_value,
        'left_indices': best_left,
        'right_indices': best_right,
        'mutual_info': best_mutual_info
    }
                
            


    
    
    

