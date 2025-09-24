import pandas as pd
from typing import List, Optional, Dict, Any

from .func_calculations import (
    calc_entropy,
    calc_conditional_entropy,
    calc_mutual_information,
    calc_majority_vote,
)

class Node:
    """
    Node class that will form the basis of our decision
    tree. 
    """
    def __init__(self):
        self.left = None
        self.right = None
        self.attr = None        # Feature to split node on
        self.edge = None        # Node parent stats
        self.vote = None        # Majority vote for classification

        self.num_positive = 0   # Number of positive samples at node
        self.num_negative = 0   # Number of negative samples at node

"""
Building decision tree
"""
def best_split(
    data:       pd.DataFrame, 
    features:   List[str], 
    target:     str, 
    threshold:  float=0.0,
    verbose:    bool=False
    )           -> Optional[Dict[str, Any]]:

    """
    Find the best feature to split on using mutual information
    Returns a dictionary with best split info or None if no valid split
    """
    
    # Bookkeeping
    entropy_target = calc_entropy(data=data, target=target)
    best_feature, best_value = None, None
    best_left, best_right = None, None
    best_mutual_info = 0.0
    
    # For each feature...
    for feat in features:

        # Calculate conditional entropy for feature over full data
        cond_entropy = calc_conditional_entropy(data=data, target=target, feature=feat)
        
        # Calculate mutual information
        mutual_info = calc_mutual_information(entropy=entropy_target, cond_entropy=cond_entropy)
        
        # If verbove... print stats
        if verbose:
            print(f"\nEntropy H({target}): {entropy_target}")
            print(f"Conditional Entropy H({target}|{feat}): {cond_entropy}")
            print(f"Mutual Information I({target}; {feat}): {mutual_info}")
        
        # If current mutual information greater than best & greater than threshold...
        if mutual_info > best_mutual_info and mutual_info > threshold:
            
            # Split the dataset on this feature
            v = calc_majority_vote(data=data, attr=feat)         # Using this to split on. Its just most occuring value
            left_subset = data[data[feat] == v]
            right_subset = data[data[feat] != v]

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
  
    # If no best_feature... return none
    if best_feature is None:
        if verbose:
            print("\nNo valid split found.")
        return None
            
    return{
        'feature': best_feature,
        'value': best_value,
        'left_indices': best_left,
        'right_indices': best_right,
        'mutual_info': best_mutual_info
    }

def build_tree(
    data:               pd.DataFrame, 
    features:           Optional[List[str]] = None,
    target:             Optional[str] = None,
    max_depth:          Optional[int] = None,
    split_threshold:    float = 0.0,
    min_split_size:     int = 0,
    depth:              int = 0,
    verbose:            bool = False
    )                   -> Node:
    """
    Build a decision tree recursively.
    """
    
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
    
    # If verbose... print info
    if verbose:
        print(f"\nDepth: {depth}")
        print(f"Best split: {split}")
        print(f"Best feature: {best_feature}")
        print(f"Left indices: {left_indices}")
        print(f"Right indices: {right_indices}\n")
        
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

def print_tree(
    node:       Node, 
    indent:     str=""
    ):
    
    """
    Pretty print the decision tree.
    """
    
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

"""
Prune decision tree
"""
def grab_val_indices(
    node:       Node, 
    data:       pd.DataFrame, 
    indices:    Optional[List[int]] = None,
    out:        Optional[Dict[int, List[int]]] = None
    ) ->        Dict[int, List[int]]:

    """
    Grab validation indices for each node in the tree.
    """
    
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

def error_prune(
    node:       Node, 
    data:       pd.DataFrame, 
    target:     str, 
    indices:    Dict[int, List[int]]
    )           -> int:
    
    """
    Prune the decision tree using error-based pruning.
    """
    
    # Get indices for this node
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
Generate predictions with decision tree
"""

def predict_row(
    node:   Node, 
    row:    pd.Series
    )       -> int:
    
    """
    Generate predictions for a single row using the decision tree.
    """
    # While not at leaf...
    while node.attr is not None:
        feat, v = node.attr
        
        # Set node to be left if feature value matches, else right
        node = node.left if row[feat] == v else node.right
    return node.vote

def predict_df(
    node:   Node,
    data:   pd.DataFrame
    )       -> List[int]:
    
    """
    Generate predictions for a dataframe using the decision tree.
    """
    
    # Empty list to hold predictions
    predictions = []
    
    # For each row in the dataframe...
    for _, row in data.iterrows():
        # Predict and append to list
        predictions.append(predict_row(node, row))
    return predictions

