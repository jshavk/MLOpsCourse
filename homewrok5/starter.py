import pickle
import pandas as pd
import numpy as np
import os

def read_data(filename, categorical):
    """Read and preprocess data."""
    df = pd.read_parquet(filename)
    df['duration'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
    df['duration'] = df['duration'].dt.total_seconds() / 60
    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

# Load model
with open('homewrok5/model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

# Q1: Standard deviation (March 2023)
print("\n=== Q1: Standard Deviation (March 2023) ===")
df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet', categorical)
y_pred = model.predict(dv.transform(df[categorical].to_dict(orient='records')))
std_dev = np.std(y_pred)
print(f"Answer Q1: {std_dev:.2f}")
print(f"Closest option: {min([1.24, 6.24, 12.28, 18.28], key=lambda x: abs(x - std_dev))}")

# Q2: Output file size
print("\n=== Q2: Output File Size ===")
df['ride_id'] = '2023/03_' + df.index.astype('str')
y_pred = model.predict(dv.transform(df[categorical].to_dict(orient='records')))
df_result = pd.DataFrame({'ride_id': df['ride_id'], 'predicted_duration': y_pred})
os.makedirs('output', exist_ok=True)
output_file = 'output/predictions_2023-03.parquet'
df_result.to_parquet(output_file, engine='pyarrow', compression=None, index=False)
file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
print(f"Answer Q2: {file_size_mb:.1f}M")
print(f"Closest option: {min([36, 46, 56, 66], key=lambda x: abs(x - file_size_mb))}M")

# Q3
print("\n=== Q3: Notebook to Script ===")
print("Answer Q3: jupyter nbconvert --to script")

# Q4
print("\n=== Q4: Pipenv Hash ===")
print("Answer Q4: [Check your Pipfile.lock file]")

# Q5: Mean duration (April 2023)
print("\n=== Q5: Mean Duration (April 2023) ===")
df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-04.parquet', categorical)
y_pred = model.predict(dv.transform(df[categorical].to_dict(orient='records')))
mean_dur = np.mean(y_pred)
print(f"Answer Q5: {mean_dur:.2f}")
print(f"Closest option: {min([7.29, 14.29, 21.29, 28.29], key=lambda x: abs(x - mean_dur))}")


print("\n" + "="*60)
print("SUBMIT THESE ANSWERS:")
print("="*60)
print(f"Q1: {min([1.24, 6.24, 12.28, 18.28], key=lambda x: abs(x - std_dev))}")
print(f"Q2: {min([36, 46, 56, 66], key=lambda x: abs(x - file_size_mb))}M")
print("Q3: jupyter nbconvert --to script")
print("Q4: [Your Pipfile.lock hash]")
print(f"Q5: {min([7.29, 14.29, 21.29, 28.29], key=lambda x: abs(x - mean_dur))}")
print("="*60)