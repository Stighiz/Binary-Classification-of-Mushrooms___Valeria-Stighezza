import numpy as np
import pandas as pd


def split_train_test(X, y, test_size=0.2, random_state=None, shuffle=True):
    """
    Splits the dataset into training and test sets.
    Parameters:
        X (DataFrame): feature dataset.
        y (DataFrame or Series): target values.
        test_size (float or int): fraction or number of samples for the test set.
        random_state (int): seed for reproducibility.
        shuffle (bool): if True, shuffles the data before splitting.
    Returns:
        X_train, X_test, y_train, y_test as DataFrames.
    """
    if random_state is not None:
        np.random.seed(random_state)

    num_samples = len(X)
    indices = np.arange(num_samples)


    if isinstance(test_size, float):
        test_size = int(num_samples * test_size)

    if shuffle:
        np.random.shuffle(indices)

    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    return X_train, X_test, y_train, y_test


###################################################################################


def reset_indices(l):
  return [df.reset_index(drop=True) for df in l]


###################################################################################


def calculate_class_statistics(X_train):
    """
    Calculates medians for numerical features and modes for categorical ones.
    Parameters:
        X_train (DataFrame): The training features dataset.
        X_test (DataFrame): The test features dataset.
    Returns:
        tuple: Two dictionaries containing medians and modes for each target class.

    """
    medians = {}
    modes = {}

    for col in X_train.columns:
        if X_train[col].dtype == 'object':  # Feature categorica
            new_val = X_train[col].mode()[0] if not X_train[col].mode().empty else None
            modes[col] = new_val
        else:  # Feature numerica
            new_val = X_train[col].median()
            medians[col] = new_val

    return medians, modes


###################################################################################


def fill_missing_values(X_subset, medians, modes):
    """
    Fills missing values in the dataset using medians for numerical columns and modes for categorical ones.
    Parameters:
        X_subset (DataFrame): The dataset with missing values.
        medians (dict): Dictionary of medians for numerical columns.
        modes (dict): Dictionary of modes for categorical columns.
    Returns:
        DataFrame: The dataset with missing values filled in.
    """
    for col in X_subset.columns:
        if X_subset[col].dtype == 'object':     # Categorical feature 
            X_subset[col] = X_subset[col].fillna(modes.get(col, None))
        else:                                   # Numerical feature
            X_subset[col] = X_subset[col].fillna(medians.get(col, None))
    
    return X_subset


###################################################################################


def NaN_summary(X_train, X_test, when):
  nan_summary = pd.DataFrame({
    "Dataset": ["X_train", "X_test"],
    "Number of NaN ("+when+")": [X_train.isna().sum().sum(), X_test.isna().sum().sum()]})
  return nan_summary
