import pandas as pd

def clean_claims_data(df):
    """
    Cleans raw claims data by:
    - Removing duplicates
    - Filling missing values
    - Converting date fields
    """
    df = df.drop_duplicates()
    df = df.fillna(0)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df
