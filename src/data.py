import pandas as pd
import numpy as np

def load_data(file_path): # function to load data
    df = pd.read_csv(file_path)
    return df
def check_missing(df): # check columns with missing values
    missing = df.isnull().sum()
    return missing[missing > 0]

def clean_calendar(calendar_df):
    cal = calendar_df.copy()
    #holiday is 1 but name is missing, we dont know which holiday it is
    cond = (cal['holiday'] == 1) & (cal['holiday_name'].isnull())
    
    cal.loc[cond, 'holiday_name'] = 'Unknown Holiday'
    cal['holiday_name'] = cal['holiday_name'].fillna('No Holiday')
    return cal

def clean_sales(sales_df, is_train=True):
    df = sales_df.copy()
    if is_train:
        df = df.dropna(subset=['sales', 'total_orders'], how='all')
    discount_cols = [
        'type_0_discount', 'type_1_discount', 'type_2_discount',
        'type_3_discount', 'type_4_discount', 'type_5_discount', 'type_6_discount'
    ]
    #fixing negative discounts
    for col in discount_cols:
        df.loc[df[col] < 0, col] = 0.0
        df.loc[df[col] > 1, col] = 1.0
    df['max_discount'] = df[discount_cols].max(axis=1)
    return df

def normalize_price(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()

    #min max scaling for each warehouse, min and max only from train so test uses the same scale
    min_price = train_df.groupby('warehouse')['sell_price_main'].min()
    max_price = train_df.groupby('warehouse')['sell_price_main'].max()
    train_min = train_df['warehouse'].map(min_price)

    train_max = train_df['warehouse'].map(max_price)
    train_df['price_norm'] = (train_df['sell_price_main'] - train_min) / (train_max - train_min)
    test_min = test_df['warehouse'].map(min_price)
    test_max = test_df['warehouse'].map(max_price)

    test_df['price_norm'] = (test_df['sell_price_main'] - test_min) / (test_max - test_min)
    
    return train_df, test_df


def merge_tables(sales_df, inventory_df, calendar_df): #merge sales with inventory and calendar on id and date
    inv_cols = inventory_df[['unique_id', 'product_unique_id', 'L1_category_name_en', 'L2_category_name_en']]
    cal_cols = calendar_df[['date', 'warehouse', 'holiday', 'shops_closed', 'winter_school_holidays', 'school_holidays']]
    df = sales_df.merge(inv_cols, on='unique_id', how='left')
    df = df.merge(cal_cols, on=['date', 'warehouse'], how='left')
    return df