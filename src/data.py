import pandas as pd
import numpy as np

def load_data(file_path): # function to load data
    df = pd.read_csv(file_path)
    return df
