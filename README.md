# DSE-Intro-Refactor
Homework for Assignment #5

Refactoring an assignment I recently submitted for COSC 522. The original version required everything in a single .py file, resulting in a large, monolithic, and hard-to-maintain script. The code was written quickly and is in a raw, unrefined state.

Some folders are automatically created so if you plan to run the refactored code, make sure you're cd'd into the project directory.

## Usage

```bash
python decision_tree.py <dataset_prefix> [max_depth] [threshold] [min_split_size] [verbose]
```

**Arguments:**

- `<dataset_prefix>`: Prefix for dataset files (e.g., `dataset` for `dataset_train.tsv` and `dataset_val.tsv`) **(required)**
- `[max_depth]`: Maximum depth of the tree (default: `None` for unlimited)
- `[threshold]`: Minimum information gain required to split (default: `0.0`)
- `[min_split_size]`: Minimum number of samples required to split (default: `0`)
- `[verbose]`: Verbosity flag (`True`/`False`, default: `False`)

At minimum, provide the dataset prefix as a command-line argument.

Below is a breakdown of key differences between the original and refactored versions of my assignment. Generated using GPT 4.1 and revised by me.

## Directory Structure

- `/original`: Contains the original, single-file implementation.
- `/refactored`: Contains the improved, modularized version split across multiple files and folders.

## Key Differences Between `/original` and `/refactored`

### 1. Modularization
- **Original**: All logic (data loading, calculations, tree building, prediction, pruning, etc.) is in a single script (`decision_tree.py`).
- **Refactored**: Code is split into logical modules:
	- `func_calculations.py`: Core calculation functions (entropy, mutual information, error rate, etc.)
	- `func_tree.py`: Decision tree logic (Node class, tree building, pruning, prediction)
	- `func_utils.py`: Utility functions (data loading, etc.)
	- `decision_tree.py`: Main script for running experiments, now much cleaner and focused on workflow.

### 2. Readability & Maintainability
- **Original**: Large blocks of code, minimal separation of concerns, hard to follow or debug.
- **Refactored**: Functions and classes are clearly separated by purpose, with docstrings and type hints for clarity. Each file has a focused responsibility.

### 3. Reusability
- **Original**: Functions are defined inside `__main__`, making reuse or testing difficult.
- **Refactored**: All major logic is in importable modules, making it easy to reuse, test, or extend components.

### 4. Usability & Flexibility
- **Original**: Hardcoded paths and parameters, limited command-line flexibility.
- **Refactored**: Root directory is obtained automatically based on current working directory. Main script supports command-line arguments for dataset prefix, tree depth, thresholds, etc. Output directories are created as needed.

### 5. Documentation & Comments
- **Original**: Sparse comments, limited documentation.
- **Refactored**: Each function and class includes docstrings. The main script includes usage instructions and parameter explanations.

### 6. Evaluation & Output
- **Original**: Evaluation and output logic is mixed with core algorithm code.
- **Refactored**: Evaluation, prediction, and output are handled in the main script, keeping modules focused on computation.

---

**Summary:**

The refactored version is easier to read, maintain, and extend. It follows best practices for Python project structure, making it more suitable for future development or adaptation to new tasks.

