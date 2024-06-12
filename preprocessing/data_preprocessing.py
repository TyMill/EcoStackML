import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def clean_data(df):
    """Basic data cleaning such as handling missing values."""
    df = df.dropna()
    return df

def normalize_data(df, columns):
    """Normalize specified columns of the dataframe."""
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def preprocess_data(file_path, columns):
    """Load, clean, and normalize data."""
    df = load_data(file_path)
    df = clean_data(df)
    df = normalize_data(df, columns)
    return df
