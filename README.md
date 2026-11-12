# Rohlik Sales Forecasting - Group 51

CENG313 Introduction to Data Science term project.
Competition: https://www.kaggle.com/competitions/rohlik-sales-forecasting-challenge-v2

Members: Nisanur Çelebi - 24118080069 (GitHub: KtoxTheNisa)

## Dataset

The dataset is from Rohlik Group, an online grocery delivery service. It covers 7 warehouses in 3 countries (Czechia, Hungary and Germany). The goal is to predict daily sales (sales_hat) for each inventory item for the 14-day test period (2024-06-03 to 2024-06-16). It consists of five files:
- `sales_train.csv`: Historical daily sales, total orders, availability, selling prices, and 7 discount types (`4,007,419` rows).
- `sales_test.csv`: The 14-day forecasting period without sales and availability (47,021 rows).
- `inventory.csv`: Product categories (`L1` to `L4`) and product IDs (`unique_id` and `product_unique_id`) for each warehouse item (`5,432` rows).
- `calendar.csv`: Daily holiday and school holiday flags for each warehouse (`23,016` rows).
- `test_weights.csv`: Product weights used to calculate the competition metric, Weighted Mean Absolute Error (WMAE) (`5,390` rows).

## Files

    src/
      __init__.py
      data.py         loading, cleaning, price normalization, merging
      models.py       baseline models
      utils.py        train/val split, metrics, submission
    submissions/      Kaggle late-submission score screenshots
    data/             dataset files (excluded via .gitignore)
    assignment1_eda_baseline.ipynb
    assignment2_solution.ipynb
    requirements.txt
    README.md

## How to Download Data & Run

- **Google Colab:** Open `assignment1_eda_baseline.ipynb` in Colab ([Open in Colab](https://colab.research.google.com/github/ceng313-2026/GROUP51_CENG313/blob/main/assignment1_eda_baseline.ipynb) | [View on nbviewer](https://nbviewer.org/github/ceng313-2026/GROUP51_CENG313/blob/main/assignment1_eda_baseline.ipynb)). Add your Kaggle API token to Colab secrets as `KAGGLE_API_TOKEN`; the setup cells automatically download and unzip the competition data into `data/`, and the notebook runs from top to bottom.
- **Local:** Accept the competition rules on the [Kaggle competition page](https://www.kaggle.com/competitions/rohlik-sales-forecasting-challenge-v2/data), download the files (or run `kaggle competitions download -c rohlik-sales-forecasting-challenge-v2` and extract into `data/`), install dependencies with `pip install -r requirements.txt`, and run `assignment1_eda_baseline.ipynb`.

## Assignment 1: EDA and Baselines

**Cleaning.** We removed 52 rows with missing sales and total orders, filled missing holiday names with 'Unknown Holiday' in the calendar and clipped the discount columns to [0, 1] since a few values were negative. Prices are in different currencies for each country, so we added a normalized price per warehouse (`price_norm`). Then we merged sales with the inventory and calendar tables. We used the full training data, 4,007,367 rows.

**What we saw in the data.** Sales are right skewed, the mean is 108 but the median is only 39.5. Across categories from `inventory.csv`, `Fruit and vegetable` has the most rows and highest median sales (48.8), while `Bakery` has the highest mean sales (121.8). Total daily sales increase over the years and drop sharply around Christmas. Fridays have the highest sales. Holidays have lower average sales than normal days. Discounts are interesting: discounted products sell more in general, but small discounts (1-15%) are actually below no-discount products, and the real jump comes after 30%. However, different discount groups contain different products, so when we compare each product with its own average sales, relative sales increase across all discount levels. Checking the 7 discount columns one by one also showed that `type_0_discount` (2.61) and `type_2_discount` (2.20) boost sales the most.

**Validation.** The test period is the 14 days after the training data ends (2024-06-03 to 2024-06-16), so we kept the last 14 days of the training data for validation. We used WMAE with the test weights as the main metric.

**Results.**

| Model | WMAE | MAE | RMSE |
|---|---|---|---|
| Historical Mean | 33.25 | 41.32 | 140.86 |
| Historical Median | 32.76 | 41.38 | 149.46 |
| Ridge Regression | 74.92 | 113.13 | 397.69 |
| Recent Mean (28 days) | 31.27 | 34.87 | 111.13 |
| **Recent Median (28 days)** | **26.77** | **31.50** | **113.58** |

Using the last 28 days worked much better than the full history because sales grow over time. Between the two 28-day baselines, the recent median achieved the best WMAE (26.77) because it is less affected by big discount peaks than the mean. Ridge was the worst because it has no information about which product it is predicting, only price, discount and calendar features. None of these baselines change their prediction from day to day, so the weekly pattern is not used yet.

## Kaggle Submission History

- **2026-11-08:** Submitted `submission_baseline.csv` (`Recent Mean - 28 days` baseline) — Validation WMAE: `31.2676`, Public Score: `32.88808`, Private Score: `31.83556`.
- **2026-11-09:** Submitted `submission_recent_median.csv` (`Recent Median - 28 days` baseline) — Validation WMAE: `26.7680`, Public Score: `28.87416`, Private Score: `28.26044`. Both Kaggle scores improved by ~3.6–4.0 points and closely match our validation results, confirming that our validation setup works well.
- Score screenshots are stored in `submissions/` (`kaggle_submission1.png`, `kaggle_submission2.png`).

## Progress Logs

- **2026-10-16:** First commit, setting up the suggested repository layout (`src/`, notebooks, and `README.md`).
- **2026-10-21:** Cleared initial Colab test code and switched to local VS Code environment.
- **2026-10-28:** Removed old local experiment and switched workflow back to Google Colab to run on the full dataset.
- **2026-11-05:** Added `load_data` function to `src/data.py`.
- **2026-11-06:** Added initial EDA plots, missing value checks, and discount/outlier inspection in `assignment1_eda_baseline.ipynb`.
- **2026-11-06:** Added data cleaning (`clean_calendar`, `clean_sales`), warehouse price normalization (`normalize_price`), and table merging (`merge_tables`) functions to `src/data.py`.
- **2026-11-08:** Added train/validation split (`split_train_val`), evaluation metrics (`calculate_metrics`), and historical mean/median baseline functions (`src/utils.py`, `src/models.py`).
- **2026-11-08:** Added `predict_recent_mean_baseline` (28 days) and `predict_ridge_baseline` to `src/models.py`, and expanded EDA plots in the notebook.
- **2026-11-08:** Updated `clean_calendar` (`'Unknown Holiday'` handling) and `normalize_price` (scaling test prices using train min/max) in `src/data.py`.
- **2026-11-08:** Completed full notebook run on the entire dataset (`4,007,367` rows), submitted `submission_baseline.csv` to Kaggle, and wrote initial `README.md`.
- **2026-11-09:** Updated `README.md` with dataset overview, Kaggle submission history, and progress logs; added `inventory` category and discount type checks, `predict_recent_median_baseline`, and removed `availability` from `predict_ridge_baseline`.

## Plan for Assignment 2

Add day of week and month features, lag and rolling features for each product, and try tree based models.
