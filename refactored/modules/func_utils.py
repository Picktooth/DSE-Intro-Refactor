import pandas as pd
import numpy as np

from typing import List, Optional, Tuple, Union

from .func_tree import (
    build_tree,
    predict_df
)

from .func_calculations import (
    calc_error_rate
)

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from a file into a pandas DataFrame, given a filepath.
    """
    
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

def optim_threshold(
    thresholds: List[float], 
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    MAX_DEPTH: Optional[int] = None
):
    
    # Bookkeeping for best threshold and accuracy
    best_accuracy = 0.0
    best_threshold = 0.0
    
    # For each threshold...
    for threshold in thresholds:

            # Build tree with threshold
            tree = build_tree(train_data, max_depth=MAX_DEPTH, split_threshold=threshold)
            
            # Predictions on validation set and save to file
            val_predictions = predict_df(tree, val_data)
            val = val_data.copy()
            val['predictions'] = val_predictions
            
            # Feed dataframe with val predictions, name of target column, and name of prediction column to calculate error rate
            val_error_rate = calc_error_rate(
                val,
                target=val_data.columns[-1],
                prediction='predictions'
            )

            val_error_rate = val_error_rate * 100
            val_accuracy = (100 - val_error_rate)
        
            # If accuracy is best so far... store accuracy and threshold
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_threshold = threshold
            
            # Else, if accuracy ties best but threshold is smaller... store threshold
            elif val_accuracy == best_accuracy and threshold < best_threshold:
                best_threshold = threshold
                
    return best_threshold, best_accuracy

def optim_min_split_size(
    min_split_sizes: List[int], 
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    MAX_DEPTH: Optional[int] = None
):
    # Bookkeeping for best min split size and accuracy
    best_accuracy = 0.0
    best_min_split_size = 0

    # For each min split size...
    for min_split_size in min_split_sizes:

            # Build tree with min split size
            tree = build_tree(train_data, max_depth=MAX_DEPTH, min_split_size=min_split_size)
            # Predictions on validation set and save to file
            val_predictions = predict_df(tree, val_data)
            val = val_data.copy()
            val['predictions'] = val_predictions
            
            # Feed dataframe with val predictions, name of target column, and name of prediction column to calculate error rate
            val_error_rate = calc_error_rate(
                val,
                target=val_data.columns[-1],
                prediction='predictions'
            )

            val_error_rate = val_error_rate * 100
            val_accuracy = (100 - val_error_rate)
        
            # If accuracy is best so far... store accuracy and threshold
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_min_split_size = min_split_size

            # Else, if accuracy ties best but min split size is smaller... store min split size
            elif val_accuracy == best_accuracy and min_split_size < best_min_split_size:
                best_min_split_size = min_split_size
    
    return best_min_split_size, best_accuracy