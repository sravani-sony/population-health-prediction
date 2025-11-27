def create_features(df):
    """
    Generates patient-level features from claims data:
    - Total cost per patient
    - Total number of claims
    - Average claim cost
    - Chronic condition count (if available)
    """
    df['total_cost'] = df.groupby('patient_id')['cost'].transform('sum')
    df['claim_count'] = df.groupby('patient_id')['claim_id'].transform('count')
    df['avg_cost'] = df.groupby('patient_id')['cost'].transform('mean')

    return df
