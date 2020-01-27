# Export timeseries data for a single symbol to csv for later processing
import config
import pandas as pd
from alpha_vantage.timeseries import TimeSeries


def export_dataset(symbol):
    ts = TimeSeries(key=config.ALPHA_VANTAGE_API_KEY, output_format='pandas')
    data, meta_data = ts.get_daily(symbol, outputsize='full')

    data.to_csv(f'./{symbol}_daily.csv')


def process_dataset(path):
    data = pd.read_csv(path)
    return data
    # do some other shit


if __name__ == '__main__':
    export_dataset('TSLA')
