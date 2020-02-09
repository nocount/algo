# Export timeseries data for a single symbol to csv and process for use in model
import config
import pandas as pd
import numpy as np
from sklearn import preprocessing
from alpha_vantage.timeseries import TimeSeries

# Amount of days to look back when predicting the next days value
history_points = 50

def export_dataset(symbol):
    ts = TimeSeries(key=config.ALPHA_VANTAGE_API_KEY, output_format='pandas')
    data, meta_data = ts.get_daily(symbol, outputsize='full')

    data.to_csv(f'./{symbol}_daily.csv')


def process_dataset(path):
    data = pd.read_csv(path)
    data = data.drop(0, axis=0)
    data = data.drop('date', axis=1)
    data = data.values

    data_normalizer = preprocessing.MinMaxScaler()
    data_normalized = data_normalizer.fit_transform(data)

    # Calculate histories
    ohlcv_histories_normalized = np.array([data_normalized[i:i + history_points].copy() for i in range(len(data_normalized) - history_points)])
    next_day_open_values_normalized = np.array([data_normalized[:, 0][i + history_points].copy() for i in range(len(data_normalized) - history_points)])
    next_day_open_values_normalized = np.expand_dims(next_day_open_values_normalized, -1)

    next_day_open_values = np.array([data[:, 0][i + history_points].copy() for i in range(len(data) - history_points)])
    next_day_open_values = np.expand_dims(next_day_open_values, -1)

    y_normalizer = preprocessing.MinMaxScaler()
    y_normalizer.fit(next_day_open_values)
    
    # TODO: need to include technical indicators here
    
    return ohlcv_histories_normalized, next_day_open_values_normalized, next_day_open_values, y_normalizer
    # do some other shit


if __name__ == '__main__':
    export_dataset('AAPL')
    # data = process_dataset('TSLA_daily.csv')
    # print(data)