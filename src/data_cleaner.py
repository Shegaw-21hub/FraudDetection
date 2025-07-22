import pandas as pd
import numpy as np

def clean_fraud_data(df):
    """
    Performs initial cleaning on the fraud_data DataFrame.
    - Drops duplicates.
    - Converts signup_time and purchase_time to datetime.
    - Coerces ip_address to numeric and drops rows with invalid IPs.
    """
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    if len(df) < initial_rows:
        print(f"Dropped {initial_rows - len(df)} duplicate rows from Fraud_Data.")

    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])

    initial_ip_rows = len(df)
    df['ip_address'] = pd.to_numeric(df['ip_address'], errors='coerce')
    df.dropna(subset=['ip_address'], inplace=True)
    if len(df) < initial_ip_rows:
        print(f"Dropped {initial_ip_rows - len(df)} rows from Fraud_Data due to invalid IP addresses.")
    df['ip_address'] = df['ip_address'].astype(np.int64) # Convert to int64 after dropping NaNs

    return df

def clean_ip_to_country_data(df):
    """
    Performs initial cleaning on the ip_to_country DataFrame.
    - Drops duplicates.
    - Converts IP address bounds to int64.
    - Sorts by lower_bound_ip_address for efficient lookup.
    """
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    if len(df) < initial_rows:
        print(f"Dropped {initial_rows - len(df)} duplicate rows from IpAddress_to_Country.")

    df['lower_bound_ip_address'] = df['lower_bound_ip_address'].astype(np.int64)
    df['upper_bound_ip_address'] = df['upper_bound_ip_address'].astype(np.int64)
    df = df.sort_values(by='lower_bound_ip_address').reset_index(drop=True)
    return df

def clean_credit_card_data(df):
    """
    Performs initial cleaning on the creditcard DataFrame.
    - Drops duplicates.
    """
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    if len(df) < initial_rows:
        print(f"Dropped {initial_rows - len(df)} duplicate rows from CreditCard_Data.")
    return df
