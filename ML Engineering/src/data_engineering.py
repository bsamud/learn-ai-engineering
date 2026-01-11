"""
Data Engineering Helper Functions
This module provides reusable functions for data preprocessing and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


def load_data(filepath, nrows=None):
    """
    Load data from CSV file.

    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    nrows : int, optional
        Number of rows to load (useful for large datasets)

    Returns:
    --------
    pd.DataFrame
        Loaded dataframe
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, nrows=nrows)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def check_missing_values(df):
    """
    Check for missing values in the dataframe.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    pd.DataFrame
        Summary of missing values
    """
    missing = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum().values,
        'Missing_Percentage': (df.isnull().sum().values / len(df) * 100).round(2)
    })
    missing = missing[missing['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

    if len(missing) == 0:
        print("No missing values found!")
    else:
        print(f"\nMissing values found in {len(missing)} columns:")
        print(missing.to_string(index=False))

    return missing


def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    strategy : str
        Strategy to handle missing values: 'mean', 'median', 'mode', 'drop'
    columns : list, optional
        Specific columns to apply the strategy. If None, apply to all numeric columns.

    Returns:
    --------
    pd.DataFrame
        Dataframe with missing values handled
    """
    df_copy = df.copy()

    if columns is None:
        columns = df_copy.select_dtypes(include=[np.number]).columns

    for col in columns:
        if df_copy[col].isnull().sum() > 0:
            if strategy == 'mean':
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)
            elif strategy == 'median':
                df_copy[col].fillna(df_copy[col].median(), inplace=True)
            elif strategy == 'mode':
                df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                df_copy.dropna(subset=[col], inplace=True)

            print(f"Filled missing values in '{col}' using {strategy}")

    return df_copy


def encode_categorical(df, columns, method='onehot'):
    """
    Encode categorical variables.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        List of categorical columns to encode
    method : str
        Encoding method: 'onehot' or 'label'

    Returns:
    --------
    pd.DataFrame
        Dataframe with encoded columns
    dict (for label encoding)
        Dictionary of label encoders
    """
    df_copy = df.copy()
    encoders = {}

    for col in columns:
        if method == 'onehot':
            # One-hot encoding
            dummies = pd.get_dummies(df_copy[col], prefix=col, drop_first=True)
            df_copy = pd.concat([df_copy, dummies], axis=1)
            df_copy.drop(col, axis=1, inplace=True)
            print(f"One-hot encoded '{col}' into {len(dummies.columns)} columns")

        elif method == 'label':
            # Label encoding
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))
            encoders[col] = le
            print(f"Label encoded '{col}' ({len(le.classes_)} unique values)")

    if method == 'label':
        return df_copy, encoders
    return df_copy


def scale_features(X_train, X_test, method='standard', columns=None):
    """
    Scale numerical features.

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Testing features
    method : str
        Scaling method: 'standard' or 'minmax'
    columns : list, optional
        Specific columns to scale. If None, scale all numeric columns.

    Returns:
    --------
    tuple
        (X_train_scaled, X_test_scaled, scaler)
    """
    if columns is None:
        columns = X_train.select_dtypes(include=[np.number]).columns

    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("method must be 'standard' or 'minmax'")

    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()

    X_train_copy[columns] = scaler.fit_transform(X_train[columns])
    X_test_copy[columns] = scaler.transform(X_test[columns])

    print(f"Scaled {len(columns)} features using {method} scaling")

    return X_train_copy, X_test_copy, scaler


def create_train_test_split(X, y, test_size=0.2, random_state=42, stratify=None):
    """
    Split data into training and testing sets.

    Parameters:
    -----------
    X : pd.DataFrame or np.array
        Features
    y : pd.Series or np.array
        Target variable
    test_size : float
        Proportion of data for testing (default: 0.2)
    random_state : int
        Random seed for reproducibility
    stratify : array-like, optional
        If not None, data is split in a stratified fashion

    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test


def remove_outliers(df, columns, method='iqr', threshold=1.5):
    """
    Remove outliers from specified columns.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        Columns to check for outliers
    method : str
        Method to detect outliers: 'iqr' (Interquartile Range)
    threshold : float
        IQR multiplier (default: 1.5)

    Returns:
    --------
    pd.DataFrame
        Dataframe with outliers removed
    """
    df_copy = df.copy()
    original_size = len(df_copy)

    for col in columns:
        if method == 'iqr':
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            outliers = ((df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)).sum()
            df_copy = df_copy[(df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)]

            if outliers > 0:
                print(f"Removed {outliers} outliers from '{col}'")

    removed = original_size - len(df_copy)
    if removed > 0:
        print(f"\nTotal rows removed: {removed} ({removed/original_size*100:.2f}%)")

    return df_copy


def get_feature_types(df):
    """
    Identify numerical and categorical features in the dataframe.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    dict
        Dictionary with 'numerical' and 'categorical' column lists
    """
    numerical = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"Numerical features ({len(numerical)}): {numerical}")
    print(f"Categorical features ({len(categorical)}): {categorical}")

    return {
        'numerical': numerical,
        'categorical': categorical
    }


def save_processed_data(df, filepath):
    """
    Save processed dataframe to CSV.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to save
    filepath : str
        Output filepath
    """
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")
    print(f"Shape: {df.shape}")
