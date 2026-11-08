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
    most_common = cal[cal['holiday'] == 1]['holiday_name'].mode()[0] #fill missing holiday_name if it is actually a holiday
    cond = (cal['holiday'] == 1) & (cal['holiday_name'].isnull())
    
    cal.loc[cond, 'holiday_name'] = most_common

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
def normalize_price(df):

    df = df.copy()
    #min max scaling for each warehouse
    min_price = df.groupby('warehouse')['sell_price_main'].transform('min')
    max_price = df.groupby('warehouse')['sell_price_main'].transform('max')
    df['price_norm'] = (df['sell_price_main'] - min_price) / (max_price - min_price)
    return df


def merge_tables(sales_df, inventory_df, calendar_df): #merge sales with inventory and calendar on id and date
    inv_cols = inventory_df[['unique_id', 'product_unique_id', 'L1_category_name_en', 'L2_category_name_en']]
    cal_cols = calendar_df[['date', 'warehouse', 'holiday', 'shops_closed', 'winter_school_holidays', 'school_holidays']]
    df = sales_df.merge(inv_cols, on='unique_id', how='left')
    df = df.merge(cal_cols, on=['date', 'warehouse'], how='left')
    return df