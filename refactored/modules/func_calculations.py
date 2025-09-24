import numpy as np
import pandas as pd

from typing import Any

def calc_majority_vote(
    data:   pd.DataFrame, 
    attr:   str
    )       -> Any:

    """
    Calculate the majority vote for a given attribute in the dataset.
    """
    
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
def calc_entropy(
    data:   pd.DataFrame,
    target: str
    )       -> float:

    """
    Calculate the entropy of the target attribute in the dataset.
    """

    # Get unique values and their probabilities
    values = set(data[target])
    probs = {val: np.sum(data[target] == val) / len(data) for val in values}

    # Calculate entropy using the formula
    entropy = sum(-p * np.log2(p) for p in probs.values() if p > 0)

    return entropy

def calc_conditional_entropy(
    data:       pd.DataFrame, 
    target:     str,
    feature:    str
    )           -> float:
    """
    Calculate the conditional entropy of the target attribute given a feature.
    """
    
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

def calc_mutual_information(
    entropy:        float, 
    cond_entropy:   float
    )               -> float:
    """
    Calculate mutual information given entropy and conditional entropy.
    """
    return entropy - cond_entropy

def calc_error_rate(
    data:       pd.DataFrame, 
    target:     str,
    prediction: str
    )           -> float:
    
    """
    Calculate the error rate between the target and prediction columns in the dataset.
    """
    
    # If no data... return 0
    if len(data) == 0:
        return 0.0
    
    # Get number of incorrect predictions by comparing values in target and prediction columns
    incorrect = int((data[target] != data[prediction]).sum())
    
    # Return error rate; We'll use this to compute accuracy % as well
    return incorrect / len(data)
