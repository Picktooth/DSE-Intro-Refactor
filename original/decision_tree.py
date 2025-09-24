import os, sys, glob
import pandas as pd
import numpy as np

class Node:
    """
    Here is an arbitrary Node class that will form the basis of your decision
    tree. 
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired 
    """
    def __init__(self):
        self.left = None
        self.right = None
        self.attr = None        # Feature to split node on
        self.edge = None        # Node parent stats
        self.vote = None        # Majority vote for classification

        self.num_positive = 0   # Number of positive samples at node
        self.num_negative = 0   # Number of negative samples at node

if __name__ == '__main__':
    
    """
    Functions for calculations
    """
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
                
    """
    Functions for building, printing, predicting, pruning decision tree
    """
    def build_tree(data, features=None, target=None, max_depth=None, split_threshold=0.0, min_split_size=0, depth=0):
        
        # Grab features and target if not provided
        if features is None:
            features = data.columns[:-1].tolist()   
        if target is None:
            target = data.columns[-1]

        # Majority vote of target for current node
        node = Node()
        node.vote = calc_majority_vote(data, target)
        node.num_positive = int((data[target] == 1).sum())
        node.num_negative = int((data[target] == 0).sum())
        
        # Edge cases
        if data.empty:                          # If dataset is empty
            return node
        if len(data[target].unique()) == 1:     # If all target values are the same
            return node
        if max_depth is not None and depth >= max_depth:   # If max depth reached
            return node
        if len(data) <= min_split_size:        # If dataset too small to split
            return node
        
        
        # Splitting; if no good split, return node
        split = best_split(data, features, target, split_threshold)
        if split is None:
            return node
        
        # Unpack split info
        best_feature = split['feature']
        value = split['value']
        left_indices = split['left_indices']
        right_indices = split['right_indices']
        
        # print(f"\nBest split: {split}")
        # print(f"Best feature: {best_feature}")
        # print(f"Left indices: {left_indices}")
        # print(f"Right indices: {right_indices}\n")
        
        # Split data and repeat
        node.attr = (best_feature, value)
        # print(f"Splitting on {best_feature} = {value}\n")
        left_data = data.loc[left_indices]
        right_data = data.loc[right_indices]

        # Obtain new set of features to build subtrees on; build subtrees
        remaining_features = [f for f in features if f != best_feature]
        node.left = build_tree(left_data, remaining_features, target, max_depth, split_threshold, depth + 1)
        node.right = build_tree(right_data, remaining_features, target, max_depth, split_threshold, depth + 1)


        return node
    
    def print_tree(node, indent=""):
        # If the node is a leaf...
        if node.attr is None:
            return
        
        # If node is root...
        if indent == "":
            print(f"[NEG: {node.num_negative} | POS: {node.num_positive}]")
            
        # Otherwise, print the feature and value for children nodes
        feat, v = node.attr
        
        # If right child exists...
        if node.right is not None:
            print(f"{indent}{feat} != {v}: [NEG: {node.right.num_negative} | POS: {node.right.num_positive}]")
            print_tree(node.right, indent + "  | ")
            
        # If left child exists...
        if node.left is not None:
            print(f"{indent}{feat} = {v}: [NEG: {node.left.num_negative} | POS: {node.left.num_positive}]")
            print_tree(node.left, indent + "  | ")


    def predict_row(node, row):
        # Predictions on a single row
        
        # While not at leaf...
        while node.attr is not None:
            feat, v = node.attr
            
            # Set node to be left if feature value matches, else right
            node = node.left if row[feat] == v else node.right
        return node.vote
    
    def predict_df(node, data):
        # Predictions on a dataframe
        predictions = []
        
        # For each row in the dataframe...
        for _, row in data.iterrows():
            
            # Predict and append to list
            predictions.append(predict_row(node, row))
        return predictions
    
    def grab_val_indices(node, data, indices=None, out=None):
        
        # If out is None... initialize empty dict
        if out is None:
            out = {}

        # If indices is None... initialize using all data indices
        if indices is None:
            indices = data.index.tolist()
        
        out[id(node)] = indices
        
        # If leaf node... return
        if node.attr is None:
            return out

        # Grab feature and value to split on
        feat, v = node.attr
        
        # Build left and right subsets
        subset = data.loc[indices]
        left_indices = subset[subset[feat] == v].index.tolist()
        right_indices = subset[subset[feat] != v].index.tolist()
        
        # Recurse on left and right children
        grab_val_indices(node.left, data, left_indices, out)
        grab_val_indices(node.right, data, right_indices, out)
        return out
    
    def error_prune(node, data, target, indices):

        idx = indices.get(id(node), [])
        
        # If we don't have any indices for this node... prune
        if len(idx) == 0:
            node.left = None
            node.right = None
            node.attr = None
            return 0
        
        # If we are at a leaf... calculate error
        if node.attr is None:
            return int((data.loc[idx, target] != node.vote).sum())
        
        # Recurse on children
        left_error = error_prune(node.left, data, target, indices)
        right_error = error_prune(node.right, data, target, indices)
        subtree_error = left_error + right_error
        
        # Calculated leaf error
        leaf_error = int((data.loc[idx, target] != node.vote).sum())
        
        # If leaf error is less than subtree error... prune
        if leaf_error <= subtree_error:
            node.left = None
            node.right = None
            node.attr = None
            return leaf_error
        # Else... return subtree error
        else:
            return subtree_error
        

    """
    Main script to load data, build tree, predict, prune, and evaluate
    """
    # Loading paths
    ROOT_ = 'C:\\Users\\John\\Documents\\2.CODE\\VSCI\\Codebase\\git-repos\\MLA1'
    DATA_ = os.path.join(ROOT_, 'data')
    EVAL_ = os.path.join(ROOT_, 'eval')
    if not os.path.exists(EVAL_):
        os.makedirs(EVAL_)

    DATASET_PREFIX_ = 'heart'
    TRAIN_ = os.path.join(DATA_, f'{DATASET_PREFIX_}_train.tsv')
    VAL_ = os.path.join(DATA_, f'{DATASET_PREFIX_}_val.tsv')
    assert os.path.exists(TRAIN_), f"File {TRAIN_} does not exist"
    assert os.path.exists(VAL_), f"File {VAL_} does not exist"

    # Hyperparameters
    MAX_DEPTH = 3
    THRESHOLD = 0.00
    MIN_SPLIT_SIZE = 0
    
    # Ensure files exist, ;load if so
    assert os.path.exists(TRAIN_), f"File {TRAIN_} does not exist"
    assert os.path.exists(VAL_), f"File {VAL_} does not exist"
    train_data = load_data(TRAIN_)
    val_data = load_data(VAL_)

    # Build tree and print
    learned_tree = build_tree(train_data, max_depth=MAX_DEPTH, split_threshold=THRESHOLD, min_split_size=MIN_SPLIT_SIZE)
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
    
    print(f"\nTraining Error rate: {train_error_rate:.4f}")
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Validation Error rate: {val_error_rate:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%\n")

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
    
    print(f"\nPruned Training Error rate: {pruned_train_error_rate:.4f}")
    print(f"Pruned Training Accuracy: {pruned_train_accuracy:.2f}%")
    print(f"Pruned Error rate: {pruned_val_error_rate:.4f}")
    print(f"Pruned Accuracy: {pruned_val_accuracy:.2f}%\n")

    """
    Below you'll find code snippets for finding optimal hyperparameters
    """
    
    """
    Uncomment below to run threshold optimization
    """
    
    # Designing thresholds from 0.0 to 1.0 with step size of 0.01
    thresholds = np.arange(0.0, 1, 0.01)
    
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
                
    print(f"Best threshold: {best_threshold} w/ accuracy: {best_accuracy:.2f}%\n")

    """
    Uncomment below to run min_split_size optimization
    """                
    
    # Designating split sizes from 0 to len(train_data) with step size of 1
    min_split_sizes = np.arange(0, len(train_data), 1)
    
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
    print(f"Best min split size: {best_min_split_size} w/ accuracy: {best_accuracy:.2f}%\n")

                
                
