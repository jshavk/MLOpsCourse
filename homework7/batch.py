import os
import pickle
import pandas as pd


def get_input_path(year, month):
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = 's3://nyc-duration-prediction-alexey/taxi_type=yellow/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)


def prepare_data(df, categorical):
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    
    df['duration'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
    df['duration'] = df['duration'].dt.total_seconds() / 60
    
    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)]
    df = df.dropna(subset=categorical)
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def read_data(filename, categorical):
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')
    
    if S3_ENDPOINT_URL:
        options = {
            'client_kwargs': {
                'endpoint_url': S3_ENDPOINT_URL
            }
        }
        df = pd.read_parquet(filename, storage_options=options)
    else:
        df = pd.read_parquet(filename)
    
    df = prepare_data(df, categorical)
    
    return df


def main(year, month):
    categorical = ['PULocationID', 'DOLocationID']
    
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)
    
    print(f"Processing {year}-{month:02d}")
    print(f"Input: {input_file}")
    
    df = read_data(input_file, categorical)
    print(f"Rows: {len(df)}")
    
    with open('models/lin_reg.bin', 'rb') as f:
        dv, model = pickle.load(f)
    
    X = dv.transform(df[categorical].to_dict(orient='records'))
    y_pred = model.predict(X)
    
    print(f"Mean prediction: {y_pred.mean():.2f}")
    
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype(str)
    df_result = pd.DataFrame({
        'ride_id': df['ride_id'],
        'predicted_duration': y_pred
    })
    
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')
    
    if S3_ENDPOINT_URL:
        options = {
            'client_kwargs': {
                'endpoint_url': S3_ENDPOINT_URL
            }
        }
        df_result.to_parquet(
            output_file,
            engine='pyarrow',
            compression=None,
            index=False,
            storage_options=options
        )
    else:
        df_result.to_parquet(
            output_file,
            engine='pyarrow',
            compression=None,
            index=False
        )
    
    print(f"Results saved to {output_file}")
    
    return df_result


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) == 3:
        year = int(sys.argv[1])
        month = int(sys.argv[2])
    else:
        year = 2023
        month = 3
    
    main(year, month)