"""
MLOps Zoomcamp Homework - Week 1
Duration Prediction Model using NYC Yellow Taxi Data

Questions:
Q1. Number of columns in January data
Q2. Standard deviation of trip duration
Q3. Fraction of records after removing outliers
Q4. Dimensionality of one-hot encoded features
Q5. RMSE on training data
Q6. RMSE on validation data
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("MLOps ZOOMCAMP HOMEWORK - WEEK 1: DURATION PREDICTION")
print("="*80)

# ============================================================================
# Q1: DOWNLOADING THE DATA - How many columns?
# ============================================================================

print("\n" + "="*80)
print("Q1: DOWNLOADING THE DATA - Number of Columns")
print("="*80)

# Load January 2023 data
print("\nLoading January 2023 data...")
df_train = pd.read_parquet('data/yellow_tripdata_2023-01.parquet')

print(f"\nDataFrame shape: {df_train.shape}")
print(f"Number of rows: {df_train.shape[0]:,}")
print(f"Number of columns: {df_train.shape[1]}")

print(f"\nColumn names:")
for i, col in enumerate(df_train.columns, 1):
    print(f"  {i:2d}. {col}")

# Answer for Q1
num_columns = df_train.shape[1]
print(f"\n✓ ANSWER Q1: {num_columns} columns")

# ============================================================================
# Q2: COMPUTING DURATION - Standard Deviation
# ============================================================================

print("\n" + "="*80)
print("Q2: COMPUTING DURATION - Standard Deviation")
print("="*80)

# Convert to datetime
print("\nConverting timestamps to datetime...")
df_train['tpep_pickup_datetime'] = pd.to_datetime(df_train['tpep_pickup_datetime'])
df_train['tpep_dropoff_datetime'] = pd.to_datetime(df_train['tpep_dropoff_datetime'])

# Calculate duration in minutes
print("Calculating duration in minutes...")
df_train['duration'] = (df_train['tpep_dropoff_datetime'] - df_train['tpep_pickup_datetime']).dt.total_seconds() / 60

print(f"\nDuration statistics (before removing outliers):")
print(f"  Count: {df_train['duration'].count():,}")
print(f"  Mean: {df_train['duration'].mean():.2f} minutes")
print(f"  Median: {df_train['duration'].median():.2f} minutes")
print(f"  Std Dev: {df_train['duration'].std():.2f} minutes")
print(f"  Min: {df_train['duration'].min():.2f} minutes")
print(f"  Max: {df_train['duration'].max():.2f} minutes")

# Answer for Q2
std_duration = df_train['duration'].std()
print(f"\n✓ ANSWER Q2: Standard deviation = {std_duration:.2f} minutes")

# ============================================================================
# Q3: DROPPING OUTLIERS - Fraction of Records Remaining
# ============================================================================

print("\n" + "="*80)
print("Q3: DROPPING OUTLIERS - Fraction Remaining (1-60 minutes)")
print("="*80)

print(f"\nOriginal dataset size: {len(df_train):,} rows")

# Remove outliers: keep only 1 <= duration <= 60
df_train_filtered = df_train[(df_train['duration'] >= 1) & (df_train['duration'] <= 60)]

print(f"After removing outliers: {len(df_train_filtered):,} rows")

# Calculate fraction
fraction = len(df_train_filtered) / len(df_train)
percentage = fraction * 100

print(f"\nOutliers removed: {len(df_train) - len(df_train_filtered):,} rows")
print(f"Fraction remaining: {fraction:.4f} ({percentage:.2f}%)")

# Answer for Q3
print(f"\n✓ ANSWER Q3: Fraction = {percentage:.0f}% (approximately)")

# Continue with filtered data
df_train = df_train_filtered

# ============================================================================
# Q4: ONE-HOT ENCODING - Dimensionality
# ============================================================================

print("\n" + "="*80)
print("Q4: ONE-HOT ENCODING - Feature Matrix Dimensionality")
print("="*80)

print("\nPreparing features for one-hot encoding...")

# Convert location IDs to strings
df_train['PULocationID'] = df_train['PULocationID'].astype(str)
df_train['DOLocationID'] = df_train['DOLocationID'].astype(str)

print(f"  Unique pickup locations: {df_train['PULocationID'].nunique()}")
print(f"  Unique dropoff locations: {df_train['DOLocationID'].nunique()}")

# Convert to list of dictionaries
print("\nConverting to list of dictionaries...")
train_dicts = df_train[['PULocationID', 'DOLocationID']].to_dict(orient='records')

print(f"  Number of records: {len(train_dicts):,}")
print(f"  Sample record: {train_dicts[0]}")

# Fit DictVectorizer
print("\nFitting DictVectorizer...")
dv = DictVectorizer()
X_train = dv.fit_transform(train_dicts)

print(f"\nFeature matrix shape: {X_train.shape}")
dimensionality = X_train.shape[1]

print(f"  Rows (samples): {X_train.shape[0]:,}")
print(f"  Columns (features): {X_train.shape[1]}")

# Answer for Q4
print(f"\n✓ ANSWER Q4: Dimensionality = {dimensionality} columns")

# ============================================================================
# Q5: TRAINING A MODEL - RMSE on Training Data
# ============================================================================

print("\n" + "="*80)
print("Q5: TRAINING A MODEL - RMSE on Training Data")
print("="*80)

# Prepare target variable
y_train = df_train['duration'].values

print(f"\nTraining data:")
print(f"  Samples: {len(y_train):,}")
print(f"  Features: {X_train.shape[1]}")

# Train Linear Regression
print("\nTraining Linear Regression model...")
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions
y_pred_train = lr.predict(X_train)

# Calculate RMSE
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))

print(f"\nTraining Results:")
print(f"  Model: LinearRegression()")
print(f"  Predictions on training data made: {len(y_pred_train):,}")
print(f"  Mean prediction: {y_pred_train.mean():.2f} minutes")
print(f"  Std prediction: {y_pred_train.std():.2f} minutes")
print(f"  RMSE: {rmse_train:.2f} minutes")

# Answer for Q5
print(f"\n✓ ANSWER Q5: RMSE on training = {rmse_train:.2f}")

# ============================================================================
# Q6: EVALUATING THE MODEL - RMSE on Validation Data
# ============================================================================

print("\n" + "="*80)
print("Q6: EVALUATING THE MODEL - RMSE on Validation Data")
print("="*80)

# Load February 2023 data
print("\nLoading February 2023 data...")
df_val = pd.read_parquet('data/yellow_tripdata_2023-02.parquet')

print(f"Validation data shape: {df_val.shape}")

# Process validation data the same way as training
print("\nProcessing validation data...")

# Convert to datetime
df_val['tpep_pickup_datetime'] = pd.to_datetime(df_val['tpep_pickup_datetime'])
df_val['tpep_dropoff_datetime'] = pd.to_datetime(df_val['tpep_dropoff_datetime'])

# Calculate duration
df_val['duration'] = (df_val['tpep_dropoff_datetime'] - df_val['tpep_pickup_datetime']).dt.total_seconds() / 60

# Remove outliers
df_val = df_val[(df_val['duration'] >= 1) & (df_val['duration'] <= 60)]

print(f"  After filtering: {len(df_val):,} records")

# One-hot encoding
df_val['PULocationID'] = df_val['PULocationID'].astype(str)
df_val['DOLocationID'] = df_val['DOLocationID'].astype(str)

val_dicts = df_val[['PULocationID', 'DOLocationID']].to_dict(orient='records')

# Transform using the fitted vectorizer
X_val = dv.transform(val_dicts)

print(f"  Feature matrix shape: {X_val.shape}")

# Prepare target variable
y_val = df_val['duration'].values

# Make predictions
print("\nMaking predictions on validation data...")
y_pred_val = lr.predict(X_val)

# Calculate RMSE
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

print(f"\nValidation Results:")
print(f"  Samples: {len(y_val):,}")
print(f"  Mean actual: {y_val.mean():.2f} minutes")
print(f"  Mean prediction: {y_pred_val.mean():.2f} minutes")
print(f"  RMSE: {rmse_val:.2f} minutes")

# Answer for Q6
print(f"\n✓ ANSWER Q6: RMSE on validation = {rmse_val:.2f}")

# ============================================================================
# SUMMARY OF ALL ANSWERS
# ============================================================================

print("\n" + "="*80)
print("SUMMARY - ALL HOMEWORK ANSWERS")
print("="*80)

print(f"""
Q1. Number of columns in January data:
    Answer: {num_columns}
    Options: 16, 17, 18, 19
    
Q2. Standard deviation of trip duration:
    Answer: {std_duration:.2f}
    Options: 32.59, 42.59, 52.59, 62.59
    
Q3. Fraction of records after removing outliers (1-60 min):
    Answer: {percentage:.0f}%
    Options: 90%, 92%, 95%, 98%
    
Q4. Dimensionality of one-hot encoded feature matrix:
    Answer: {dimensionality}
    Options: 2, 155, 345, 515, 715
    
Q5. RMSE on training data:
    Answer: {rmse_train:.2f}
    Options: 3.64, 7.64, 11.64, 16.64
    
Q6. RMSE on validation data:
    Answer: {rmse_val:.2f}
    Options: 3.81, 7.81, 11.81, 16.81
""")

print("="*80)
print("HOMEWORK COMPLETE!")
print("="*80)

# ============================================================================
# ADDITIONAL ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("ADDITIONAL MODEL INSIGHTS")
print("="*80)

print(f"""
Model Performance:
  - Training RMSE: {rmse_train:.2f} minutes
  - Validation RMSE: {rmse_val:.2f} minutes
  - Difference: {abs(rmse_val - rmse_train):.2f} minutes
  
Interpretation:
  - Model explains {(1 - (rmse_train/y_train.mean()))*100:.1f}% of training variance
  - Model explains {(1 - (rmse_val/y_val.mean()))*100:.1f}% of validation variance
  - Very consistent performance (minimal overfitting)
  
Data Quality:
  - Training: {len(y_train):,} records
  - Validation: {len(y_val):,} records
  - Features: {X_train.shape[1]} (one-hot encoded locations)
  - Outliers removed: ~{(1-fraction)*100:.1f}% of original data
""")

print("\nModel Coefficients:")
print(f"  - Intercept: {lr.intercept_:.2f} minutes")
print(f"  - Number of feature weights: {len(lr.coef_)}")
print(f"  - Top 5 positive weights (longest duration increase):")

# Get top 5 positive coefficients
feature_names = dv.get_feature_names_out()
top_idx = np.argsort(lr.coef_)[-5:][::-1]
for rank, idx in enumerate(top_idx, 1):
    print(f"    {rank}. {feature_names[idx]}: +{lr.coef_[idx]:.2f} minutes")

print(f"\n  - Top 5 negative weights (shortest duration):")
bottom_idx = np.argsort(lr.coef_)[:5]
for rank, idx in enumerate(bottom_idx, 1):
    print(f"    {rank}. {feature_names[idx]}: {lr.coef_[idx]:.2f} minutes")