import pandas as pd

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