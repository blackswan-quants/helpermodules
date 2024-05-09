# Libraries used
import datetime as dt
import os
import pandas as pd
from twelvedata import TDClient
from dotenv import load_dotenv
import re
from helpermodules.memory_handling import PickleHelper

class DataFrameHelper:
    def __init__(self, filename, link, months, interval):
        self.filename = filename
        self.link = link
        self.months = months
        self.interval = interval
        self.dataframe = []
        self.tickers = []

    def load(self):
        """
        Load a DataFrame of stock dataframe from a pickle file if it exists, otherwise create a new DataFrame.

        Parameters: Obj
            self

        Returns: None
        """

        if not re.search("^.*\.pkl$", self.filename):
            self.filename += ".pkl"

        file_path = "./pickle_files/" + self.filename

        if os.path.isfile(file_path):
            self.dataframe = PickleHelper.pickle_load(self.filename).obj
            self.tickers = self.dataframe.columns.tolist()
        else:
            self.tickers = self.get_stockex_tickers()
            self.dataframe = self.loaded_df()
        return None

    def get_stockex_tickers(self):
        """
        Retrieves ticker symbols from a Wikipedia page containing stock exchange information.

        Parameters:
            self

        Returns:
            List[str]: List of ticker symbols.
        """
        tables = pd.read_html(self.link)
        df = tables[4]
        df.drop(['Company', 'GICS Sector', 'GICS Sub-Industry'],
                axis=1, inplace=True)
        tickers = df['Ticker'].values.tolist()
        return tickers

    def get_stock_data(start_date, output_size, ticker, interval):
        
        load_dotenv()
        API_KEY = os.getenv('API_KEY')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
        td = TDClient(apikey = API_KEY)

        data_to_load_tmp = output_size
        tmp_data = start_date
        parts = []

        for i in range(1, (output_size // 5000)+1):
            if (data_to_load_tmp >= 5000):
                ts = td.time_series(
                    symbol=ticker,
                    interval=interval,
                    outputsize=5000,
                    end_date = tmp_data,
                    timezone="America/New_York",
                ).as_pandas()
                data_to_load_tmp -= 5000
                parts.append(ts)
                tmp_data = parts[i-1].index.tolist()[-1] - dt.timedelta(minutes=1)
            if (data_to_load_tmp < 5000):
                ts = td.time_series(
                    symbol=ticker,
                    interval=interval,
                    outputsize=data_to_load_tmp,
                    end_date = tmp_data,
                    timezone="America/New_York",
                ).as_pandas()
                parts.append(ts)

        return pd.concat(parts)


    def loaded_df(self):
        """
        Downloads stock price data for the specified number of years and tickers using yfinance.
        Returns a pandas DataFrame and pickles the data.

        Parameters:
            years (int): Number of years of historical data to load.
            tickers (List[str]): List of ticker symbols.
            interval (str): Time frequency of historical data to load with format: ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1W', '1M' or '1Q').

        Returns:
            pandas.DataFrame: DataFrame containing downloaded stock price data.
        """
        load_dotenv()

        API_KEY = os.getenv('API_KEY')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              

        td = TDClient(apikey = API_KEY)

        stocks_dict = {}
        time_window = 21 * self.months
        start_date = dt.datetime.now().replace(hour=16, minute=0 , second=0 ,microsecond=0) + dt.timedelta(days = -2)
        for i, ticker in enumerate(self.tickers):
            print('Getting {} ({}/{})'.format(ticker, i, len(self.tickers)))
            #FIXME: add dataframe concatenation algorithm or simply pass the list of tickers, dict will be discarted
            dataframe = self.get_stock_data(start_date, time_window * 390, ticker, self.interval)
            stocks_dict[ticker] = dataframe['close']

        stocks_dataframe = pd.DataFrame.from_dict(stocks_dict)

        return stocks_dataframe

    def clean_df(self, percentage):
        """
        Cleans the DataFrame by dropping stocks with NaN values exceeding the given percentage threshold.
        The cleaned DataFrame is pickled after the operation.

        Parameters:
        self
        percentage : float
            Percentage threshold for NaN values. If greater than 1, it's interpreted as a percentage (e.g., 5 for 5%).
        
        Returns:
        None
        """
        if percentage > 1:
            percentage = percentage / 100

        for ticker in self.tickers:
            nan_values = self.dataframe[ticker].isnull().values.any()
            if nan_values:
                count_nan = self.dataframe[ticker].isnull().sum()
                if count_nan > (len(self.dataframe) * percentage):
                    self.dataframe.drop(ticker, axis=1, inplace=True)

        self.dataframe.fillna(method='ffill', inplace=True)
        #FIXME: fml this doesn't work if i have consecutive days
        PickleHelper(obj=self.dataframe).pickle_dump(filename='cleaned_nasdaq_dataframe')
