import pandas as pd
import numpy as np

def load_data(file_path): # function to load data
    df = pd.read_csv(file_path)
    return df
def check_missing(df): # check columns with missing values
    missing = df.isnull().sum()
    return missing[missing > 0]