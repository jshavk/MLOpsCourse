"""
Duration Prediction Model
This script loads taxi trip data, preprocesses it, and trains machine learning models
to predict trip duration.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 1. DATA LOADING AND EXPLORATION
# ============================================================================

def load_data(filepath):
    """Load parquet data with minimal processing"""
    print(f"Loading data from {filepath}...")
    df = pd.read_parquet(filepath)
    return df

def explore_data(df, name="Dataset"):
    """Display basic information about the dataset"""
    print(f"\n{'='*60}")
    print(f"{name} Statistics")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nData types:")
    print(df.dtypes)
    print(f"\nMissing values:")
    print(df.isnull().sum())
    print(f"\nBasic statistics:")
    print(df.describe())

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================

def preprocess_data(df, is_train=True):
    """
    Preprocess the dataframe:
    - Create duration column from dropoff - pickup time
    - Remove outliers (duration < 1 or > 60 minutes)
    - Keep only specific columns
    - Create PU_DO feature (combination of pickup and dropoff locations)
    """
    df = df.copy()
    
    # Convert timestamp columns to datetime
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    
    # Calculate duration in minutes
    df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    
    # Remove outliers
    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)]
    
    print(f"Rows after outlier removal: {len(df)}")
    
    # Keep only needed columns
    df = df[['tpep_pickup_datetime', 'trip_distance', 'PULocationID', 'DOLocationID', 'duration']]
    
    # Create combined feature
    df['PU_DO'] = df['PULocationID'].astype(str) + '_' + df['DOLocationID'].astype(str)
    
    return df

# ============================================================================
# 3. FEATURE ENGINEERING AND VECTORIZATION
# ============================================================================

def prepare_features(df_train, df_val, categorical_features, numerical_features):
    """
    Convert DataFrames to feature vectors using DictVectorizer
    """
    dv = DictVectorizer()
    
    # Convert to dictionary format
    train_dicts = df_train[categorical_features + numerical_features].to_dict(orient='records')
    val_dicts = df_val[categorical_features + numerical_features].to_dict(orient='records')
    
    # Fit on training data and transform both
    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)
    
    return X_train, X_val, dv

def prepare_target(df_train, df_val, target_column='duration'):
    """Extract target variable"""
    y_train = df_train[target_column].values
    y_val = df_val[target_column].values
    return y_train, y_val

# ============================================================================
# 4. MODEL TRAINING AND EVALUATION
# ============================================================================

def train_linear_regression(X_train, y_train, X_val, y_val):
    """Train Linear Regression model"""
    print(f"\n{'='*60}")
    print("Training Linear Regression Model")
    print(f"{'='*60}")
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    y_pred = lr.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    print(f"RMSE: {rmse:.4f}")
    
    return lr, rmse

def train_lasso(X_train, y_train, X_val, y_val, alpha=0.01):
    """Train Lasso model with L1 regularization"""
    print(f"\n{'='*60}")
    print(f"Training Lasso Model (alpha={alpha})")
    print(f"{'='*60}")
    
    lasso = Lasso(alpha=alpha, random_state=42)
    lasso.fit(X_train, y_train)
    
    y_pred = lasso.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    print(f"RMSE: {rmse:.4f}")
    
    return lasso, rmse

def save_model(model, dv, filepath='models/lin_reg.bin'):
    """Save model and vectorizer to file"""
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f_out:
        pickle.dump((dv, model), f_out)
    
    print(f"Model saved to {filepath}")

# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("\n" + "="*60)
    print("DURATION PREDICTION MODEL PIPELINE")
    print("="*60)
    
    # Configuration
    # For local testing, you can use sample data
    # Replace these paths with your actual data locations
    
    try:
        # Load data
        # Adjust these paths based on where your data is stored
        train_file = 'data/yellow_tripdata_2023-01.parquet'
        val_file = 'data/yellow_tripdata_2023-02.parquet'
        
        print("\nAttempting to load training data...")
        df_train = load_data(train_file)
        print("✓ Training data loaded successfully")
        
        print("\nAttempting to load validation data...")
        df_val = load_data(val_file)
        print("✓ Validation data loaded successfully")
        
    except FileNotFoundError as e:
        print(f"\n⚠ Data files not found: {e}")
        print("\nTo use this script, you need:")
        print("1. Create a 'data' folder in your project directory")
        print("2. Download taxi trip data (parquet files) from NYC Taxi & Limousine Commission")
        print("3. Place them in the data folder as:")
        print("   - data/yellow_tripdata_2023-01.parquet")
        print("   - data/yellow_tripdata_2023-02.parquet")
        print("\nAlternatively, you can download sample data from:")
        print("https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page")
        print("\nScript structure is ready to use once data is available.")
        return
    
    # Explore original data
    explore_data(df_train, "Training Data")
    explore_data(df_val, "Validation Data")
    
    # Preprocess
    print(f"\n{'='*60}")
    print("Preprocessing Training Data")
    print(f"{'='*60}")
    df_train = preprocess_data(df_train, is_train=True)
    
    print(f"\n{'='*60}")
    print("Preprocessing Validation Data")
    print(f"{'='*60}")
    df_val = preprocess_data(df_val, is_train=False)
    
    # Feature engineering
    categorical_features = ['PU_DO']
    numerical_features = ['trip_distance']
    
    print(f"\n{'='*60}")
    print("Feature Engineering")
    print(f"{'='*60}")
    print(f"Categorical features: {categorical_features}")
    print(f"Numerical features: {numerical_features}")
    
    X_train, X_val, dv = prepare_features(df_train, df_val, categorical_features, numerical_features)
    y_train, y_val = prepare_target(df_train, df_val)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    
    # Train models
    lr_model, lr_rmse = train_linear_regression(X_train, y_train, X_val, y_val)
    lasso_model, lasso_rmse = train_lasso(X_train, y_train, X_val, y_val, alpha=0.01)
    
    # Compare models
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"Linear Regression RMSE: {lr_rmse:.6f}")
    print(f"Lasso (alpha=0.01) RMSE: {lasso_rmse:.6f}")
    
    if lr_rmse < lasso_rmse:
        print(f"✓ Linear Regression performs better")
        best_model = lr_model
    else:
        print(f"✓ Lasso model performs better")
        best_model = lasso_model
    
    # Save best model
    save_model(best_model, dv, filepath='models/duration_predictor.bin')
    
    print(f"\n{'='*60}")
    print("Pipeline completed successfully!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()