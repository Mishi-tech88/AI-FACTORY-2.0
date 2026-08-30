# src/preprocess.py
import pandas as pd
import numpy as np
from datetime import datetime

# ------------------ 1. LOAD RAW DATA ------------------
df_prod = pd.read_csv('data/raw/production.csv', parse_dates=['timestamp'])
df_sensor = pd.read_csv('data/raw/sensors.csv', parse_dates=['timestamp'])
df_images = pd.read_csv('data/raw/image_metadata.csv')
df_text = pd.read_csv('data/raw/maintenance_notes.csv', parse_dates=['date'])

# ------------------ 2. CLEANING ------------------
# Drop duplicates
df_prod.drop_duplicates(inplace=True)
df_sensor.drop_duplicates(inplace=True)

# Handle missing values (use direct assignment, not inplace on chained index)
df_prod['production_count'] = df_prod['production_count'].fillna(df_prod['production_count'].median())
df_prod['quality_score'] = df_prod['quality_score'].fillna(df_prod['quality_score'].median())
df_prod['temperature'] = df_prod['temperature'].fillna(df_prod['temperature'].median())

df_sensor.sort_values(['machine_id','timestamp'], inplace=True)
# Forward fill per machine
df_sensor[['vibration','pressure','rpm']] = df_sensor.groupby('machine_id')[['vibration','pressure','rpm']].ffill().bfill()

# Ensure correct data types
df_prod['timestamp'] = pd.to_datetime(df_prod['timestamp'])
df_sensor['timestamp'] = pd.to_datetime(df_sensor['timestamp'])
df_prod['machine_id'] = df_prod['machine_id'].astype('category')
df_sensor['machine_id'] = df_sensor['machine_id'].astype('category')

# Outlier capping (IQR method)
def cap_outliers(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower, upper)
    return df

cap_outliers(df_prod, ['production_count','quality_score','temperature'])
cap_outliers(df_sensor, ['vibration','pressure','rpm'])

# ------------------ 3. MERGE (asof) ------------------
# Important: sort by both machine_id and timestamp, and reset index
df_prod_sorted = df_prod.sort_values('timestamp')
df_sensor_sorted = df_sensor.sort_values('timestamp')

# Merge as-of: for each production timestamp, get the nearest sensor reading (same machine)
df_merged = pd.merge_asof(
    df_prod_sorted,
    df_sensor_sorted,
    on='timestamp',
    by='machine_id',
    direction='nearest'
)
df_merged.dropna(inplace=True)

# ------------------ 4. FEATURE ENGINEERING ------------------
# Rolling features (avoid leakage: shift(1) before rolling)
for col in ['vibration','pressure','rpm']:
    df_merged[f'{col}_rolling_mean_24'] = df_merged.groupby('machine_id')[col].transform(
        lambda x: x.shift(1).rolling(24, min_periods=1).mean()
    )
    df_merged[f'{col}_rolling_std_24'] = df_merged.groupby('machine_id')[col].transform(
        lambda x: x.shift(1).rolling(24, min_periods=1).std()
    )
    df_merged[f'{col}_lag1'] = df_merged.groupby('machine_id')[col].shift(1)
    df_merged[f'{col}_lag24'] = df_merged.groupby('machine_id')[col].shift(24)

# One-hot encoding categoricals
df_merged = pd.get_dummies(df_merged, columns=['machine_id','operator'], drop_first=True)

# ------------------ 5. TRAIN/VAL/TEST SPLIT (temporal) ------------------
df_merged = df_merged.sort_values('timestamp').reset_index(drop=True)
split_date_train = df_merged['timestamp'].quantile(0.70)
split_date_val = df_merged['timestamp'].quantile(0.85)

train = df_merged[df_merged['timestamp'] <= split_date_train]
val = df_merged[(df_merged['timestamp'] > split_date_train) & (df_merged['timestamp'] <= split_date_val)]
test = df_merged[df_merged['timestamp'] > split_date_val]

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

# ------------------ 6. SAVE PROCESSED DATA ------------------
train.to_csv('data/processed/train.csv', index=False)
val.to_csv('data/processed/val.csv', index=False)
test.to_csv('data/processed/test.csv', index=False)

print("Preprocessing complete. Files saved in data/processed/")