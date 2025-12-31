import os
import pandas as pd
from datetime import datetime
import boto3


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def test_integration():
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL', 'http://localhost:4566')
    BUCKET = 'nyc-duration'
    
    options = {
        'client_kwargs': {
            'endpoint_url': S3_ENDPOINT_URL
        }
    }
    
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]
    
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df_input = pd.DataFrame(data, columns=columns)
    
    input_file = f's3://{BUCKET}/in/2023-01.parquet'
    
    print(f"Saving test data to {input_file}")
    df_input.to_parquet(
        input_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )
    
    s3_client = boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT_URL,
        region_name='us-east-1',
        aws_access_key_id='test',
        aws_secret_access_key='test'
    )
    
    response = s3_client.head_object(Bucket=BUCKET, Key='in/2023-01.parquet')
    file_size = response['ContentLength']
    print(f"File size: {file_size} bytes")
    
    os.environ['INPUT_FILE_PATTERN'] = f's3://{BUCKET}/in/{{year:04d}}-{{month:02d}}.parquet'
    os.environ['OUTPUT_FILE_PATTERN'] = f's3://{BUCKET}/out/{{year:04d}}-{{month:02d}}.parquet'
    os.environ['S3_ENDPOINT_URL'] = S3_ENDPOINT_URL
    
    print("Running batch.py 2023 1")
    os.system('python batch.py 2023 1')
    
    output_file = f's3://{BUCKET}/out/2023-01.parquet'
    
    print(f"Reading predictions from {output_file}")
    df_predictions = pd.read_parquet(output_file, storage_options=options)
    
    print(f"Predictions:\n{df_predictions}")
    
    total_duration = df_predictions['predicted_duration'].sum()
    print(f"Sum: {total_duration:.2f}")
    
    assert len(df_predictions) > 0
    assert 'predicted_duration' in df_predictions.columns


if __name__ == '__main__':
    test_integration()