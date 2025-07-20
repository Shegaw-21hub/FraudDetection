import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

# Assume the functions from eda.ipynb are available or can be imported.
# For simplicity in testing, we'll mock or re-implement necessary parts.

# Mock ip_to_country_sorted DataFrame for get_country testing
mock_ip_ranges_df = pd.DataFrame({
    'lower_bound_ip_address': [0, 1000, 2000, 3000],
    'upper_bound_ip_address': [999, 1999, 2999, 3999],
    'country': ['CountryA', 'CountryB', 'CountryC', 'CountryD']
}).sort_values(by='lower_bound_ip_address').reset_index(drop=True)

def get_country(ip_address, ip_ranges_df):
    """
    Mock of the get_country function from eda.ipynb for testing.
    """
    idx = ip_ranges_df['lower_bound_ip_address'].searchsorted(ip_address, side='right') - 1
    if idx >= 0 and idx < len(ip_ranges_df) and ip_address <= ip_ranges_df.loc[idx, 'upper_bound_ip_address']:
        return ip_ranges_df.loc[idx, 'country']
    return 'Unknown'

def calculate_time_since_signup(signup_time, purchase_time):
    """
    Mock of the time_since_signup_seconds calculation.
    """
    time_diff = (purchase_time - signup_time).total_seconds()
    return max(0, time_diff)

def calculate_transaction_diffs_and_counts(df):
    """
    Mock of the transaction frequency and velocity calculations.
    Assumes df has 'user_id', 'device_id', 'purchase_time'
    """
    # First sort for user_id based diffs and counts
    df = df.sort_values(by=['user_id', 'purchase_time']).reset_index(drop=True)
    df['time_diff_user'] = df.groupby('user_id')['purchase_time'].diff().dt.total_seconds()
    df['time_diff_user'] = df['time_diff_user'].fillna(0) # Corrected inplace operation

    df['user_transaction_count'] = df.groupby('user_id')['purchase_time'].transform('count')

    # Second sort for device_id based diffs and counts
    # Note: This re-sorts the DataFrame, affecting the order of previously calculated columns
    df = df.sort_values(by=['device_id', 'purchase_time']).reset_index(drop=True)
    df['time_diff_device'] = df.groupby('device_id')['purchase_time'].diff().dt.total_seconds()
    df['time_diff_device'] = df['time_diff_device'].fillna(0) # Corrected inplace operation

    df['device_transaction_count'] = df.groupby('device_id')['purchase_time'].transform('count')

    return df

class TestDataProcessing:

    def test_get_country_valid_ip(self):
        """Test get_country with an IP within a known range."""
        assert get_country(500, mock_ip_ranges_df) == 'CountryA'
        assert get_country(1500, mock_ip_ranges_df) == 'CountryB'
        assert get_country(2500, mock_ip_ranges_df) == 'CountryC'

    def test_get_country_boundary_ip(self):
        """Test get_country with IPs at the boundaries of ranges."""
        assert get_country(0, mock_ip_ranges_df) == 'CountryA'
        assert get_country(999, mock_ip_ranges_df) == 'CountryA'
        assert get_country(1000, mock_ip_ranges_df) == 'CountryB'
        assert get_country(1999, mock_ip_ranges_df) == 'CountryB'

    def test_get_country_unknown_ip(self):
        """Test get_country with IPs outside any known range."""
        assert get_country(4000, mock_ip_ranges_df) == 'Unknown'
        assert get_country(-1, mock_ip_ranges_df) == 'Unknown'
        assert get_country(10000, mock_ip_ranges_df) == 'Unknown'

    def test_remove_duplicates(self):
        """Test duplicate removal functionality."""
        data = {'col1': [1, 2, 1, 3], 'col2': ['a', 'b', 'a', 'c']}
        df = pd.DataFrame(data)
        df_cleaned = df.drop_duplicates()
        assert len(df_cleaned) == 3
        assert df_cleaned.iloc[0]['col1'] == 1
        assert df_cleaned.iloc[1]['col1'] == 2
        assert df_cleaned.iloc[2]['col1'] == 3

    def test_datetime_conversion(self):
        """Test conversion of string timestamps to datetime objects."""
        data = {'signup_time': ['2023-01-01 10:00:00'], 'purchase_time': ['2023-01-01 11:00:00']}
        df = pd.DataFrame(data)
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        assert isinstance(df['signup_time'].iloc[0], datetime)
        assert isinstance(df['purchase_time'].iloc[0], datetime)

    def test_ip_address_coercion_and_dropna(self):
        """Test IP address type conversion and handling of invalid IPs."""
        data = {'ip_address': [12345, 'invalid_ip', 67890]}
        df = pd.DataFrame(data)
        initial_len = len(df)
        df['ip_address'] = pd.to_numeric(df['ip_address'], errors='coerce')
        df.dropna(subset=['ip_address'], inplace=True)
        # After dropna, convert to int64 if all remaining values are non-NaN and can be integers
        # This mirrors the behavior in eda.ipynb
        df['ip_address'] = df['ip_address'].astype(np.int64)
        assert len(df) == initial_len - 1 # One row dropped
        assert df['ip_address'].dtype == np.int64 # Should be int after conversion and dropna

    def test_time_since_signup_seconds(self):
        """Test calculation of time_since_signup_seconds."""
        signup = datetime(2023, 1, 1, 10, 0, 0)
        purchase_later = datetime(2023, 1, 1, 10, 5, 0) # 300 seconds later
        purchase_earlier = datetime(2023, 1, 1, 9, 50, 0) # 10 minutes earlier

        assert calculate_time_since_signup(signup, purchase_later) == 300.0
        assert calculate_time_since_signup(signup, purchase_earlier) == 0.0 # Should be 0 if purchase is before signup

    def test_transaction_diffs_and_counts(self):
        """Test calculation of time_diff_user/device and transaction counts."""
        data = {
            'user_id': [1, 1, 2, 2, 1],
            'device_id': ['A', 'A', 'B', 'B', 'C'],
            'purchase_time': [
                datetime(2023, 1, 1, 10, 0, 0),
                datetime(2023, 1, 1, 10, 30, 0),
                datetime(2023, 1, 1, 11, 0, 0),
                datetime(2023, 1, 1, 11, 15, 0),
                datetime(2023, 1, 1, 10, 45, 0) # User 1, new device
            ]
        }
        df = pd.DataFrame(data)
        df_processed = calculate_transaction_diffs_and_counts(df.copy())

        # Expected values for time_diff_user after the *final* sort by device_id
        # Original indices: 0, 1, 2, 3, 4
        # Sorted by user_id, purchase_time: (0), (1), (4), (2), (3)
        # time_diff_user values: [0.0, 1800.0, 900.0, 0.0, 900.0]
        # Then sorted by device_id, purchase_time:
        # (0, A, 10:00) -> time_diff_user 0.0
        # (1, A, 10:30) -> time_diff_user 1800.0
        # (2, B, 11:00) -> time_diff_user 0.0
        # (3, B, 11:15) -> time_diff_user 900.0
        # (4, C, 10:45) -> time_diff_user 900.0
        expected_time_diff_user = pd.Series([0.0, 1800.0, 0.0, 900.0, 900.0], name='time_diff_user') # Added name
        pd.testing.assert_series_equal(df_processed['time_diff_user'], expected_time_diff_user, check_dtype=False)


        # Test time_diff_device
        expected_time_diff_device_values = pd.Series([0.0, 1800.0, 0.0, 900.0, 0.0], name='time_diff_device') # Added name
        pd.testing.assert_series_equal(df_processed['time_diff_device'], expected_time_diff_device_values, check_dtype=False)


        # Test device_transaction_count
        expected_device_counts = pd.Series([2, 2, 2, 2, 1], name='device_transaction_count') # Added name
        pd.testing.assert_series_equal(df_processed['device_transaction_count'], expected_device_counts, check_dtype=False)

        # Test user_transaction_count
        expected_user_counts = pd.Series([3, 3, 2, 2, 3], name='user_transaction_count') # Added name
        pd.testing.assert_series_equal(df_processed['user_transaction_count'], expected_user_counts, check_dtype=False)
