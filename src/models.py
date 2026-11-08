import pandas as pd
import numpy as np
from src.utils import calculate_metrics


def predict_mean_baseline(train_df, target_df): #mean sales per product
    product_means = train_df.groupby('unique_id')['sales'].mean()
    global_mean = train_df['sales'].mean()
    
    preds = target_df['unique_id'].map(product_means).fillna(global_mean)
    return preds.values


def predict_median_baseline(train_df, target_df):
    #median sales per product
    product_medians = train_df.groupby('unique_id')['sales'].median()
    global_median = train_df['sales'].median()
    
    preds = target_df['unique_id'].map(product_medians).fillna(global_median)
    return preds.values



def train_baseline_models(train_part, val_part):
    val_weights = val_part['weight'].values
    
    y_val = val_part['sales'].values
    
    pred_mean = predict_mean_baseline(train_part, val_part)
    pred_median = predict_median_baseline(train_part, val_part)
    res_mean = calculate_metrics(y_val, pred_mean, val_weights)
    res_median = calculate_metrics(y_val, pred_median, val_weights)
    results_df = pd.DataFrame([res_mean, res_median], index=['Historical Mean', 'Historical Median'])
    
    return results_df, pred_mean, pred_median