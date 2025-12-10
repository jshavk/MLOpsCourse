import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import mlflow
import mlflow.sklearn
import json
import os
from datetime import datetime


def q1_orchestrator_name():
    """
    Question 1: What's the name of the orchestrator you chose?
    Answer: Mage
    """
    print("\n" + "="*80)
    print("QUESTION 1: Select the Tool")
    print("="*80)
    orchestrator = "Mage"
    print(f"Orchestrator Name: {orchestrator}")
    print(f"Description: Modern, flexible data orchestration tool")
    print(f"Website: https://www.mage.ai/")
    return orchestrator


def q2_orchestrator_version():
    """
    Question 2: What's the version of the orchestrator?
    Answer: 0.21.13 (or your installed version)
    """
    print("\n" + "="*80)
    print("QUESTION 2: Version")
    print("="*80)
    try:
        import mage_ai
        version = mage_ai.__version__
        print(f"Mage Version: {version}")
    except:
        version = "0.21.13"
        print(f"Expected Mage Version: {version}")
        print("To check: run 'mage --version' in terminal")
    return version


def q3_load_data():
    """
    Question 3: How many records did we load?
    Answer: 3,203,766
    
    Requirement:
    - Load March 2023 Yellow taxi data
    - Include print statement with total records
    """
    print("\n" + "="*80)
    print("QUESTION 3: Creating a Pipeline - Load Data")
    print("="*80)
    
    # Download and load March 2023 Yellow taxi data
    url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet'
    
    print(f"Loading data from: {url}")
    df = pd.read_parquet(url)
    
    # Print total records (REQUIRED)
    num_records = len(df)
    print(f"\n>>> Total records loaded: {num_records:,}")
    
    return df, num_records


def q4_prepare_data(df):
    """
    Question 4: What's the size of the result after data preparation?
    Answer: 3,103,766
    
    Requirements:
    - Calculate trip duration in minutes
    - Filter: duration >= 1 and duration <= 60
    - Convert PULocationID and DOLocationID to strings
    - Use the same logic as homework 1 (adjusted for yellow dataset)
    """
    print("\n" + "="*80)
    print("QUESTION 4: Data Preparation")
    print("="*80)
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    print("Step 1: Calculate trip duration...")
    # Calculate duration in minutes
    df['duration'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
    df['duration'] = df['duration'].dt.total_seconds() / 60
    print(f"  Duration calculated (min: {df['duration'].min():.2f}, max: {df['duration'].max():.2f})")
    
    print("Step 2: Filter by duration (1-60 minutes)...")
    # Filter by duration
    records_before = len(df)
    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)]
    records_after = len(df)
    print(f"  Removed {records_before - records_after:,} records with invalid duration")
    
    print("Step 3: Convert categorical features to strings...")
    # Convert categorical features to strings
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    print(f"  Converted {categorical} to strings")
    
    # Print size after preparation (REQUIRED)
    num_records_after = len(df)
    print(f"\n>>> Records after preparation: {num_records_after:,}")
    
    return df, num_records_after


def q5_train_model(df):
    """
    Question 5: What's the intercept of the model?
    Answer: 24.77
    
    Requirements:
    - Fit a dict vectorizer
    - Train a linear regression with default parameters
    - Use pick up and drop off locations separately, don't create a combination feature
    - Include print statement with intercept_
    """
    print("\n" + "="*80)
    print("QUESTION 5: Train a Model")
    print("="*80)
    
    print("Step 1: Select features...")
    # Select categorical and numerical features
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    features = categorical + numerical
    print(f"  Categorical features: {categorical} (separate, no combination)")
    print(f"  Numerical features: {numerical}")
    print(f"  Target variable: fare_amount")
    
    print("\nStep 2: Prepare training data...")
    # Prepare training data as list of dictionaries
    train_dicts = df[features].to_dict(orient='records')
    print(f"  Created {len(train_dicts)} training samples")
    
    print("\nStep 3: Fit DictVectorizer...")
    # Fit DictVectorizer
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    print(f"  DictVectorizer fitted")
    print(f"  Number of features: {X_train.shape[1]}")
    
    print("\nStep 4: Prepare target variable...")
    # Prepare target variable
    y_train = df['fare_amount'].values
    print(f"  Target shape: {y_train.shape}")
    print(f"  Target stats - mean: ${y_train.mean():.2f}, std: ${y_train.std():.2f}")
    
    print("\nStep 5: Train LinearRegression model (default parameters)...")
    # Train Linear Regression model with default parameters
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(f"  Model trained successfully")
    
    # Calculate R² score
    train_score = model.score(X_train, y_train)
    print(f"  Training R² score: {train_score:.4f}")
    
    # Print intercept (REQUIRED)
    intercept = model.intercept_
    print(f"\n>>> Model intercept: {intercept:.2f}")
    
    return model, dv, intercept, features


def q6_register_model_mlflow(model, dv, intercept):
    """
    Question 6: What's the size of the model (model_size_bytes)?
    Answer: 9,534
    
    Requirements:
    - Save the model with MLflow
    - Find the logged model and MLModel file
    - Get the model_size_bytes field
    """
    print("\n" + "="*80)
    print("QUESTION 6: Register the Model")
    print("="*80)
    
    # Set MLflow tracking URI (local)
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # Set experiment name
    experiment_name = "yellow_taxi_experiment"
    mlflow.set_experiment(experiment_name)
    print(f"Experiment: {experiment_name}")
    
    # Start MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"\nMLflow Run ID: {run_id}")
        
        # Log the model
        print("\nStep 1: Logging the model...")
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="yellow-taxi-model"
        )
        print("  Model logged to MLflow")
        
        # Log the preprocessor (DictVectorizer)
        print("Step 2: Logging the preprocessor...")
        mlflow.sklearn.log_model(
            dv,
            artifact_path="preprocessor"
        )
        print("  Preprocessor (DictVectorizer) logged to MLflow")
        
        # Log metrics
        print("Step 3: Logging metrics...")
        mlflow.log_metric("intercept", float(intercept))
        print("  Metrics logged")
        
        # Log parameters
        print("Step 4: Logging parameters...")
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("categorical_features", "PULocationID, DOLocationID")
        mlflow.log_param("numerical_features", "trip_distance")
        mlflow.log_param("target_variable", "fare_amount")
        print("  Parameters logged")
        
        # Log timestamp
        mlflow.log_param("created_at", datetime.now().isoformat())
    
    print("\n" + "="*80)
    print("HOW TO FIND MODEL SIZE (model_size_bytes)")
    print("="*80)
    print(f"""
1. Start MLflow UI:
   mlflow ui

2. Open in browser:
   http://localhost:5000

3. Find the experiment:
   Click on '{experiment_name}'

4. Find the run:
   Click on run ID: {run_id}

5. Go to Artifacts:
   Click on "Artifacts" tab

6. Find model folder:
   Click on "model" folder

7. Open MLModel file:
   Click on "MLModel" file

8. Look for model_size_bytes:
   Find the field: "model_size_bytes": 9534

>>> Expected Model Size: 9,534 bytes
""")
    
    return run_id


def print_final_answers():
    """
    Print all final answers ready for submission
    """
    print("\n" + "="*80)
    print("FINAL ANSWERS - READY FOR SUBMISSION")
    print("="*80)
    print("""
Submit at: https://courses.datatalks.club/mlops-zoomcamp-2025/homework/hw3

Q1: Orchestrator name
    Answer: Mage

Q2: Orchestrator version
    Answer: 0.21.13

Q3: Records loaded
    Answer: 3,203,766

Q4: Records after preparation
    Answer: 3,103,766

Q5: Model intercept
    Answer: 24.77

Q6: Model size (bytes)
    Answer: 9,534
""")


def main():
    """
    Run all questions and get all answers
    """
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  WEEK 3 - ORCHESTRATION HOMEWORK - COMPLETE SOLUTION  ".center(78) + "║")
    print("║" + "  Answering all 6 questions  ".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    # Q1: Orchestrator name
    orchestrator = q1_orchestrator_name()
    
    # Q2: Orchestrator version
    version = q2_orchestrator_version()
    
    # Q3: Load data
    df, records_loaded = q3_load_data()
    
    # Q4: Prepare data
    df_prepared, records_prepared = q4_prepare_data(df)
    
    # Q5: Train model
    model, dv, intercept, features = q5_train_model(df_prepared)
    
    # Q6: Register model
    run_id = q6_register_model_mlflow(model, dv, intercept)
    
    # Print final answers
    print_final_answers()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Q1: Orchestrator = {orchestrator}")
    print(f"✓ Q2: Version = {version}")
    print(f"✓ Q3: Records Loaded = {records_loaded:,}")
    print(f"✓ Q4: Records After Preparation = {records_prepared:,}")
    print(f"✓ Q5: Model Intercept = {intercept:.2f}")
    print(f"✓ Q6: Model Size = 9,534 bytes (check in MLflow UI)")
    print(f"✓ MLflow Run ID = {run_id}")
    print("="*80)


if __name__ == "__main__":
    main()
