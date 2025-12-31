import pandas as pd
from datetime import datetime
from batch import prepare_data


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def test_prepare_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]
    
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)
    
    categorical = ['PULocationID', 'DOLocationID']
    df_processed = prepare_data(df, categorical)
    
    assert len(df_processed) == 1
    assert df_processed['PULocationID'].iloc[0] == '1'
    assert df_processed['DOLocationID'].iloc[0] == '1'
    
    duration = df_processed['duration'].iloc[0]
    assert 7 < duration < 9


def test_prepare_data_empty():
    df = pd.DataFrame(columns=['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime'])
    categorical = ['PULocationID', 'DOLocationID']
    result = prepare_data(df, categorical)
    assert len(result) == 0


def test_prepare_data_short_duration():
    data = [(1, 1, dt(1, 0, 0), dt(1, 0, 30))]
    df = pd.DataFrame(data, columns=['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime'])
    categorical = ['PULocationID', 'DOLocationID']
    result = prepare_data(df, categorical)
    assert len(result) == 0


def test_prepare_data_long_duration():
    data = [(1, 1, dt(1, 0), dt(2, 5))]
    df = pd.DataFrame(data, columns=['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime'])
    categorical = ['PULocationID', 'DOLocationID']
    result = prepare_data(df, categorical)
    assert len(result) == 0


def test_prepare_data_valid_middle():
    data = [(1, 1, dt(1, 0), dt(1, 30))]
    df = pd.DataFrame(data, columns=['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime'])
    categorical = ['PULocationID', 'DOLocationID']
    result = prepare_data(df, categorical)
    assert len(result) == 1
    assert 29 < result['duration'].iloc[0] < 31