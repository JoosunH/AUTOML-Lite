import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

def detect_task_type(target_series: pd.Series, verbose: bool = False) -> str:
    """
    Automatically determine if a supervised ML task is classification or regression.

    Parameters:
    -----------
    target_series : pd.Series
        The target variable (labels) as a pandas Series.
    
    verbose : bool
        If True, prints reasoning behind the detected task type.

    In here pd.series is used to represent the target variable so its easy to guess
    what the data type is.
    
    Returns:
    --------
    str
        'classification' or 'regression'
    """
    
    cleaned_series = target_series.dropna()
    
    if cleaned_series.dtype == 'object' or pd.api.types.is_categorical_dtype(cleaned_series):
        if verbose:
            print("Detected object or categorical dtype —> classification")
        return 'classification'
    
    if pd.api.types.is_bool_dtype(cleaned_series):
        if verbose:
            print("Detected boolean dtype —> classification")
        return 'classification'
    
    unique_values = cleaned_series.nunique()
    total_values = len(cleaned_series)
    
    if verbose:
        print(f"Unique values: {unique_values}, Total values: {total_values}")
        
    if unique_values <= 20 and (unique_values / total_values) < 0.1:
        if verbose:
            print("Detected low cardinality —> classification")
        return 'classification'
    
    if verbose:
        print("Detected high cardinality or continuous values —> regression")
    return 'regression'

def preprocess_data(df: pd.DataFrame, target_column: str):
    """
    Preprocess the DataFrame by handling missing values and encoding categorical variables.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame to preprocess.
    
    target_column : str
        The name of the target column in the DataFrame.

    Returns:
    --------
    pd.DataFrame
        The preprocessed DataFrame.
    """
    
    # Handle missing values
    X = df.drop(columns=[target_column])
    y_raw = df[target_column]
    
    if y_raw.dtype == 'object' or y_raw.dtype.name == 'category':
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y_raw), name=target_column)
    else:
        y = y_raw
    
    num_cols = X.select_dtypes(include=['number']).columns.to_list()
    cat_cols = X.select_dtypes(exclude=['number']).columns.to_list()
    
    num_pipeline = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())
    #mean imputer check all the numerical columns and fill the missing values with mean
    # StandardScaler It rescales your data so each feature has mean 0 and standard deviation 1. The value can be outside of 0 and 1 
    cat_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))
    #It creates a new binary (0 or 1) column for each category. (Onehotencoder)
    
    preprocessing = ColumnTransformer([('num', num_pipeline, num_cols), 
                                       ('cat', cat_pipeline, cat_cols)], 
                                      remainder='passthrough'
                                      )

    x_prepared = preprocessing.fit_transform(X)
    task_type = detect_task_type(y, verbose=True)
    
    return preprocessing, x_prepared, y, task_type
    