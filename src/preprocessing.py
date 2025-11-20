"""
Data preprocessing module for customer churn prediction
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def load_data(filepath):
    """Load customer data from CSV"""
    df = pd.read_csv(filepath)
    print(f"Loaded data shape: {df.shape}")
    return df


def feature_engineering(df):
    """
    Create engineered features from raw data
    
    Args:
        df: Raw dataframe
    Returns:
        df_fe: Dataframe with engineered features
    """
    df_fe = df.copy()
    
    # Balance per product
    df_fe['balance_per_product'] = df_fe['balance'] / (
        df_fe['products_number'].replace(0, np.nan)
    )
    df_fe['balance_per_product'].fillna(0, inplace=True)
    
    # Salary to balance ratio
    df_fe['salary_balance_ratio'] = df_fe['estimated_salary'] / (
        df_fe['balance'].replace(0, np.nan)
    )
    df_fe['salary_balance_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df_fe['salary_balance_ratio'].fillna(
        df_fe['salary_balance_ratio'].median(), inplace=True
    )
    
    # Age group
    bins = [0, 25, 35, 45, 55, 65, 100]
    labels = ['<25', '25-34', '35-44', '45-54', '55-64', '65+']
    df_fe['age_group'] = pd.cut(df_fe['age'], bins=bins, labels=labels)
    
    # Tenure bucket
    df_fe['tenure_bucket'] = pd.cut(
        df_fe['tenure'], 
        bins=[-1, 0, 2, 5, 10, 100], 
        labels=['0', '1-2', '3-5', '6-10', '10+']
    )
    
    # High balance flag
    df_fe['high_balance'] = (
        df_fe['balance'] > df_fe['balance'].quantile(0.75)
    ).astype(int)
    
    return df_fe


def get_preprocessor():
    """
    Create sklearn preprocessing pipeline
    
    Returns:
        preprocessor: ColumnTransformer with scaling and encoding
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
    """
    numeric_features = [
        'credit_score', 'age', 'tenure', 'balance', 
        'products_number', 'estimated_salary',
        'balance_per_product', 'salary_balance_ratio'
    ]
    
    categorical_features = [
        'country', 'gender', 'credit_card', 
        'active_member', 'age_group', 'tenure_bucket', 'high_balance'
    ]
    
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    return preprocessor, numeric_features, categorical_features


def prepare_features(df_fe, target='churn', drop_cols=['customer_id']):
    """
    Prepare X and y for modeling
    
    Args:
        df_fe: Feature-engineered dataframe
        target: Target column name
        drop_cols: Columns to drop
    Returns:
        X, y: Features and target
    """
    features = [c for c in df_fe.columns if c not in [target] + drop_cols]
    X = df_fe[features]
    y = df_fe[target]
    
    return X, y