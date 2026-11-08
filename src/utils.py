import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def split_train_val(df, val_days=14):
    #last 14 days for validation
    max_date = pd.to_datetime(df['date']).max()
    cutoff_date = max_date - pd.Timedelta(days=val_days)
    
    dates = pd.to_datetime(df['date'])

    train_part = df[dates <= cutoff_date].copy()

    val_part = df[dates > cutoff_date].copy()
    return train_part, val_part


def calculate_metrics(y_true, y_pred, weights=None):
#calculate wmae, mae and rmse

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    if weights is not None:
        wmae = np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)
    else:
        wmae = mae
        
    return {
        'WMAE': round(wmae, 4),
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4)
    }

#for saving
def save_submission(test_df, predictions, filename='submission_baseline.csv'):
    sub = pd.DataFrame({
        'id': test_df['unique_id'].astype(str) + '_' + test_df['date'].astype(str),
        'sales_hat': predictions
    })
    sub.to_csv(filename, index=False)
    return sub