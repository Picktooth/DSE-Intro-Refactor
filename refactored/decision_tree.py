import os, sys
import pandas as pd
import numpy as np

from modules import (
    load_data,
    build_tree,
    print_tree,
    predict_df,
    calc_error_rate,
    grab_val_indices,
    error_prune,
    optim_threshold,
    optim_min_split_size
)

if __name__ == '__main__':


    """
    Main script to load data, build tree, predict, prune, and evaluate
    
    Usage: python decision_tree.py <dataset_prefix> [max_depth] [threshold] [min_split_size] [verbose]
    - dataset_prefix: Prefix for dataset files (e.g., 'dataset' for 'dataset_train.tsv' and 'dataset_val.tsv')
    - max_depth: Maximum depth of the tree (default: None for unlimited)
    - threshold: Minimum information gain required to split (default: 0.0)
    - min_split_size: Minimum number of samples required to split (default: 0)
    - verbose: Verbosity flag (Bool, default: False)
    
    At minimum, provide dataset prefix as command line argument.
    """
    # Loading paths
    ROOT_ = os.getcwd()
    DATA_ = os.path.join(ROOT_, 'data')


    # Expect at minimum dataset prefix as command line argument
    if len(sys.argv) < 2:
        print("Usage: python decision_tree.py <dataset_prefix>")
        sys.exit(1)
    else:
        DATASET_PREFIX_ = sys.argv[1]

    EVAL_ = os.path.join(ROOT_, 'eval', DATASET_PREFIX_)
    if not os.path.exists(EVAL_):
        os.makedirs(EVAL_)
        
    TRAIN_ = os.path.join(DATA_, f'{DATASET_PREFIX_}_train.tsv')
    VAL_ = os.path.join(DATA_, f'{DATASET_PREFIX_}_val.tsv')
    assert os.path.exists(TRAIN_), f"File {TRAIN_} does not exist"
    assert os.path.exists(VAL_), f"File {VAL_} does not exist"

    # Hyperparameters; set defaults if not provided
    MAX_DEPTH =         int(sys.argv[2]) if len(sys.argv) >= 3 else None
    THRESHOLD =         float(sys.argv[3]) if len(sys.argv) >= 4 else 0.0
    MIN_SPLIT_SIZE =    int(sys.argv[4]) if len(sys.argv) >= 5 else 0
    VERBOSE =           bool(int(sys.argv[5])) if len(sys.argv) >= 6 else False
    
    # Ensure files exist, ;load if so
    assert os.path.exists(TRAIN_), f"File {TRAIN_} does not exist"
    assert os.path.exists(VAL_), f"File {VAL_} does not exist"
    train_data = load_data(TRAIN_)
    val_data = load_data(VAL_)

    # Build tree and print
    learned_tree = build_tree(train_data, max_depth=MAX_DEPTH, split_threshold=THRESHOLD, min_split_size=MIN_SPLIT_SIZE, verbose=VERBOSE)
    print(f"\nLearned Decision Tree from {os.path.basename(TRAIN_)}:")
    print_tree(learned_tree)
    
    
    # Predictions on training set and save to file
    train_predictions = predict_df(learned_tree, train_data)
    train = train_data.copy()
    train['predictions'] = train_predictions
    train.to_csv(os.path.join(EVAL_, f'{DATASET_PREFIX_}_train_predictions.tsv'), sep='\t', index=False)
    
    # Predictions on validation set and save to file
    val_predictions = predict_df(learned_tree, val_data)
    val = val_data.copy()
    val['predictions'] = val_predictions
    val.to_csv(os.path.join(EVAL_, f'{DATASET_PREFIX_}_val_predictions.tsv'), sep='\t', index=False)

    # Feed dataframe with training predictions, name of target column, and name of prediction column to calculate error rate
    train_error_rate = calc_error_rate(
        train,
        target=train_data.columns[-1],
        prediction='predictions'
    )
    train_error_rate = train_error_rate * 100
    train_accuracy = (100 - train_error_rate)
    
    # Feed dataframe with val predictions, name of target column, and name of prediction column to calculate error rate
    val_error_rate = calc_error_rate(
        val,
        target=val_data.columns[-1],
        prediction='predictions'
    )

    val_error_rate = val_error_rate * 100
    val_accuracy = (100 - val_error_rate)

    # Pruning
    indices = grab_val_indices(learned_tree, val_data)
    error_prune(learned_tree, val_data, target=val_data.columns[-1], indices=indices)
    print(f"\nPruned Decision Tree:")
    print_tree(learned_tree)
    
    # Predictions on training set after pruning and save to file
    pruned_train_predictions = predict_df(learned_tree, train_data)
    train = train_data.copy()
    train['pruned_predictions'] = pruned_train_predictions
    train.to_csv(os.path.join(EVAL_, f'{DATASET_PREFIX_}_train_pruned_predictions.tsv'), sep='\t', index=False)
    pruned_train_error_rate = calc_error_rate(
        train,
        target=train_data.columns[-1],
        prediction='pruned_predictions'
    )
    pruned_train_error_rate = pruned_train_error_rate * 100
    pruned_train_accuracy = (100 - pruned_train_error_rate)

    # Predictions on validation set after pruning and save to file
    pruned_predictions = predict_df(learned_tree, val_data)
    val = val_data.copy()
    val['pruned_predictions'] = pruned_predictions
    val.to_csv(os.path.join(EVAL_, f'{DATASET_PREFIX_}_val_pruned_predictions.tsv'), sep='\t', index=False)
    pruned_val_error_rate = calc_error_rate(
        val,
        target=val_data.columns[-1],
        prediction='pruned_predictions'
    )
    pruned_val_error_rate = pruned_val_error_rate * 100
    pruned_val_accuracy = (100 - pruned_val_error_rate)
    
    print(f"\nTraining Error rate: {train_error_rate:.4f}")
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Validation Error rate: {val_error_rate:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    
    print(f"\nPruned Training Error rate: {pruned_train_error_rate:.4f}")
    print(f"Pruned Training Accuracy: {pruned_train_accuracy:.2f}%")
    print(f"Pruned Error rate: {pruned_val_error_rate:.4f}")
    print(f"Pruned Accuracy: {pruned_val_accuracy:.2f}%")


    """
    Below you'll find code snippets for finding optimal hyperparameters
    """
    
    # Designing thresholds from 0.0 to 1.0 with step size of 0.01
    thresholds = np.arange(0.0, 1, 0.01)
    
    # Run function to find best threshold in range
    best_threshold, best_accuracy_thresh = optim_threshold(
        thresholds,
        train_data,
        val_data,
        MAX_DEPTH=MAX_DEPTH
    )
       
    
    # Designating split sizes from 0 to len(train_data) with step size of 1
    min_split_sizes = np.arange(0, len(train_data), 1)
    
    # Run function to find best min split size in range
    best_mss, best_accuracy_mss = optim_min_split_size(
        min_split_sizes,
        train_data,
        val_data,
        MAX_DEPTH=MAX_DEPTH
    )

    print(f"\nBest threshold: {best_threshold} w/ accuracy: {best_accuracy_thresh:.2f}%")
    print(f"Best min split size: {best_mss} w/ accuracy: {best_accuracy_mss:.2f}%\n")