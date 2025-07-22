import pandas as pd

def load_fraud_data(path='../data/Fraud_Data.csv'):
    """Loads the Fraud_Data.csv dataset."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: Fraud_Data.csv not found at {path}. Please check the path.")
        raise

def load_ip_to_country_data(path='../data/IpAddress_to_Country.csv'):
    """Loads the IpAddress_to_Country.csv dataset."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: IpAddress_to_Country.csv not found at {path}. Please check the path.")
        raise

def load_credit_card_data(path='../data/creditcard.csv'):
    """Loads the creditcard.csv dataset."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: creditcard.csv not found at {path}. Please check the path.")
        raise
