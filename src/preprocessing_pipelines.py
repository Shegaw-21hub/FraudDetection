from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

def get_fraud_preprocessor(numerical_features, categorical_features):
    """
    Defines and returns the preprocessing pipeline for Fraud_Data.
    """
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    print("Preprocessing pipeline for Fraud_Data defined.")
    return preprocessor

def get_credit_card_preprocessor(numerical_features):
    """
    Defines and returns the preprocessing pipeline for CreditCard_Data.
    """
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features)
        ],
        remainder='passthrough'
    )
    print("Preprocessing pipeline for CreditCard_Data defined.")
    return preprocessor
