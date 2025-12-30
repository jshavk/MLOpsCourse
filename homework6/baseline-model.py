"""
Week 5 - Homework Solution with Your Actual Data
Run this on your Mac to get answers Q1 & Q3

Usage:
    python week5_your_data.py

Make sure you have:
    pip install pandas numpy
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("\n" + "="*80)
print("WEEK 5 HOMEWORK - USING YOUR ACTUAL DATA")
print("="*80 + "\n")

# Path to your data
data_path = '/Users/julia_shavkatsishvili/Desktop/mlopsCourse/MLOpsCourse/data/green_tripdata_2024-03.parquet'

try:
    # Load data
    print("Loading your data...")
    df = pd.read_parquet(data_path)
    
    print(f"✓ Data loaded successfully!")
    
    # ========================================================================
    # Q1: SHAPE OF THE DATA
    # ========================================================================
    
    print(f"\n" + "="*80)
    print("Q1: SHAPE OF MARCH 2024 GREEN TAXI DATA")
    print("="*80)
    
    rows = df.shape[0]
    cols = df.shape[1]
    
    print(f"\nDataFrame shape: ({rows}, {cols})")
    print(f"Number of rows: {rows}")
    print(f"Number of columns: {cols}")
    
    # Find closest option
    options_q1 = [72044, 78537, 57457, 54396]
    closest_q1 = min(options_q1, key=lambda x: abs(x - rows))
    
    print(f"\nQ1 Options:")
    for opt in options_q1:
        marker = " ← CLOSEST" if opt == closest_q1 else ""
        print(f"  • {opt}{marker}")
    
    print(f"\n>>> Q1 ANSWER: {closest_q1}")
    
    # ========================================================================
    # Q2: METRIC EXPLANATION
    # ========================================================================
    
    print(f"\n" + "="*80)
    print("Q2: ADD METRIC FOR FARE_AMOUNT (QUANTILE=0.5)")
    print("="*80)
    
    if 'fare_amount' in df.columns:
        fare_median = df['fare_amount'].quantile(0.5)
        
        print(f"\nYour data contains 'fare_amount' column:")
        print(f"  Mean: {df['fare_amount'].mean():.2f}")
        print(f"  Median (Q2): {fare_median:.2f}")
        print(f"  Min: {df['fare_amount'].min():.2f}")
        print(f"  Max: {df['fare_amount'].max():.2f}")
        
        print(f"\n>>> Q2 ANSWER: ColumnQuantileMetric")
        print(f"\nExplanation:")
        print(f"  This metric calculates the 0.5 quantile (median) for a column")
        print(f"  Code: ColumnQuantileMetric(column_name='fare_amount', quantile=0.5)")
    else:
        print("WARNING: 'fare_amount' column not found!")
    
    # ========================================================================
    # Q3: DAILY QUANTILE MAXIMUM
    # ========================================================================
    
    print(f"\n" + "="*80)
    print("Q3: MAXIMUM DAILY QUANTILE 0.5 (MARCH 2024)")
    print("="*80)
    
    if 'lpep_pickup_datetime' in df.columns and 'fare_amount' in df.columns:
        # Ensure datetime format
        df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])
        
        # Extract date
        df['date'] = df['lpep_pickup_datetime'].dt.date
        
        # Calculate daily quantile 0.5 for fare_amount
        daily_quantiles = df.groupby('date')['fare_amount'].quantile(0.5)
        
        print(f"\nDaily Quantile 0.5 for fare_amount:")
        print(daily_quantiles)
        
        # Find statistics
        max_q = daily_quantiles.max()
        min_q = daily_quantiles.min()
        mean_q = daily_quantiles.mean()
        max_date = daily_quantiles.idxmax()
        min_date = daily_quantiles.idxmin()
        
        print(f"\nStatistics:")
        print(f"  Maximum: {max_q:.4f} (on {max_date})")
        print(f"  Minimum: {min_q:.4f} (on {min_date})")
        print(f"  Mean: {mean_q:.4f}")
        print(f"  Std Dev: {daily_quantiles.std():.4f}")
        print(f"  Total days: {len(daily_quantiles)}")
        
        # Find closest option
        options_q3 = [10, 12.5, 14.2, 14.8]
        closest_q3 = min(options_q3, key=lambda x: abs(x - max_q))
        
        print(f"\nQ3 Options:")
        for opt in options_q3:
            marker = " ← CLOSEST" if opt == closest_q3 else ""
            print(f"  • {opt}{marker}")
        
        print(f"\nYour max value: {max_q:.2f}")
        print(f"\n>>> Q3 ANSWER: {closest_q3}")
    else:
        print("WARNING: Required columns not found!")
    
    # ========================================================================
    # Q4: DASHBOARD CONFIG
    # ========================================================================
    
    print(f"\n" + "="*80)
    print("Q4: DASHBOARD CONFIG FILE LOCATION")
    print("="*80)
    
    print(f"""
Options:
  1. 05-monitoring
  2. 05-monitoring/config     
  3. 05-monitoring/dashboards
  4. 05-monitoring/data

>>> Q4 ANSWER: 05-monitoring/config
""")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("FINAL ANSWERS - READY TO SUBMIT")
    print("="*80)
    
    print(f"""
Q1: {closest_q1}
Q2: ColumnQuantileMetric
Q3: {closest_q3}
Q4: 05-monitoring/config

""")
    
    print("="*80)
    print("="*80 + "\n")

except FileNotFoundError:
    print(f"❌ Error: File not found at {data_path}")
    print(f"\nMake sure the file exists and the path is correct.")
    print(f"Current path: {data_path}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()