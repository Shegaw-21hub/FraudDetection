import pandas as pd
import numpy as np

def get_country(ip_address, ip_ranges_df):
    """
    Maps an IP address to a country using a sorted IP range DataFrame.
    Assumes ip_ranges_df is sorted by 'lower_bound_ip_address'.
    """
    idx = ip_ranges_df['lower_bound_ip_address'].searchsorted(ip_address, side='right') - 1
    if idx >= 0 and idx < len(ip_ranges_df) and ip_address <= ip_ranges_df.loc[idx, 'upper_bound_ip_address']:
        return ip_ranges_df.loc[idx, 'country']
    return 'Unknown'

def apply_ip_to_country_mapping(fraud_df, ip_ranges_df):
    """
    Applies the IP-to-Country mapping to the fraud_data DataFrame.
    """
    print("Applying IP to Country mapping for Fraud_Data (this may take a moment)...")
    fraud_df['country'] = fraud_df['ip_address'].apply(lambda x: get_country(x, ip_ranges_df))
    print("IP to Country mapping complete!")
    return fraud_df

def engineer_fraud_features(df):
    """
    Engineers time-based and velocity features for the fraud_data DataFrame.
    Assumes 'signup_time' and 'purchase_time' are datetime objects.
    """
    print("Extracting time-based and velocity features for Fraud_Data...")

    # Time-based features
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    df['day_of_year'] = df['purchase_time'].dt.dayofyear
    df['month_of_year'] = df['purchase_time'].dt.month
    df['week_of_year'] = df['purchase_time'].dt.isocalendar().week.astype(int)

    # Time since signup
    df['time_since_signup_seconds'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
    df['time_since_signup_seconds'] = df['time_since_signup_seconds'].apply(lambda x: max(0, x))

    # Ensure data is sorted for diff calculations
    df = df.sort_values(by=['user_id', 'purchase_time']).reset_index(drop=True)

    # Time difference for user
    df['time_diff_user'] = df.groupby('user_id')['purchase_time'].diff().dt.total_seconds()
    df['time_diff_user'] = df['time_diff_user'].fillna(0) # Handle first transaction for user

    # User transaction count
    df['user_transaction_count'] = df.groupby('user_id')['purchase_time'].transform('count')

    # Ensure data is sorted for device diff calculations (re-sorts the dataframe)
    df = df.sort_values(by=['device_id', 'purchase_time']).reset_index(drop=True)

    # Time difference for device
    df['time_diff_device'] = df.groupby('device_id')['purchase_time'].diff().dt.total_seconds()
    df['time_diff_device'] = df['time_diff_device'].fillna(0) # Handle first transaction for device

    # Device transaction count
    df['device_transaction_count'] = df.groupby('device_id')['purchase_time'].transform('count')

    print("Feature Engineering complete for Fraud_Data.")
    return df
