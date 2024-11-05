
import numpy as np
from datetime import timedelta, datetime
import os
import pandas as pd
from twelvedata import TDClient
import yfinance as yf # import yfinance 
import re
from helpermodules.memory_handling import PickleHelper
from dotenv import load_dotenv
import time

class IndexData_Retrieval:
    """
    A class for downloading and processing historical stock price data using yfinance.

    Parameters:
        filename (str): Name of the pickle file to save or load df.
        link (str): URL link to a Wikipedia page containing stock exchange information.
        interval (str): Time self.frequency of historical data to load (e.g., '1min', '1day', '1W').
        self.frequency (str): self.frequency of data intervals ('daily', 'weekly', 'monthly', etc.).
        years (int, optional): Number of years of historical data to load (default: None).
        months (int, optional): Number of months of historical data to load (default: None).

    Methods:
        getdata():
            Loads a df of stock price data from a pickle file if it exists, otherwise creates a new df.
            Returns:
                pandas.df or None: df containing stock price data if loaded successfully, otherwise None.

        get_stockex_tickers():
            Retrieves ticker symbols from a Wikipedia page containing stock exchange information.
            Returns:
                List[str]: List of ticker symbols extracted from the specified Wikipedia page.
        
        fetch_yahoo_dat():


        loaded_df():
            Downloads historical stock price data for the specified time window and tickers using yfinance.
            Returns:
                pandas.df or None: df containing downloaded stock price data if successful, otherwise None.
    """

    def __init__(self, filename, link, frequency, years=None, months=None):
        self.filename = filename
        self.link = link
        self.df = pd.DataFrame()
        self.frequency = frequency
        self.tickers = []
        self.years = years
        self.months = months

    def getdata(self):
        """
        Load a df of stock price data from a pickle file if it exists, otherwise create a new df.
        Returns:
            pandas.df or None: df containing stock price data if loaded successfully, otherwise None.
        """
        if not re.search("^.*\.pkl$", self.filename):
            self.filename += ".pkl"
        file_path = "./pickle_files/" + self.filename

        if os.path.isfile(file_path):
            self.df = PickleHelper.pickle_load(self.filename).obj
            self.tickers = self.df.columns.tolist()
            return self.df
        else:
            self.tickers = self.get_stockex_tickers()
            self.df = self.loaded_df()

        return None


    def get_stockex_tickers(self):
        """
        Retrieves ticker symbols from a Wikipedia page containing stock exchange information.
        Returns:
            List[str]: List of ticker symbols extracted from the specified Wikipedia page.
        """
        tables = pd.read_html(self.link)
        df = tables[4]
        df.drop(['Company', 'GICS Sector', 'GICS Sub-Industry'],
                axis=1, inplace=True)
        tickers = df['Ticker'].values.tolist()
        return tickers
    
    # MODIFIED BY USING YFINANCE INSTEAD OF YFINANCE
    
    def fetch_yahoo_data(self, start_date : datetime, end_date: datetime): 
        """
        Fetches historical data for multiple tickers within a specified date range from Yahoo Finance and stores the results in a DataFrame.
        This method handles Yahoo Finance's limitations on data intervals and historical periods, adjusting the start date if necessary.
        retrieves adjusted closing prices for the specified tickers and frequency.

        Parameters:
        -----------
        start_date : datetime
            The start date for fetching data.
        end_date : datetime
             end date for fetching data.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the historical adjusted closing prices for all specified tickers.
        """
        # Since Yahoo Finance has limitations on some frequency intervals and historical periods, 
        # we have added a validations to make sure the parameters are ok.

        # Valid interval of frequencies
        valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m',
                           '1h', '1d', '5d', '1wk', '1mo', '3mo']
        if self.frequency not in valid_intervals:
            raise ValueError(
                f"Frequency not valid '{self.frequency}'. Valid frequencies are: {valid_intervals}")

        # Limitations for intraday datas
        max_periods = {
            '1m': timedelta(days=7),
            '2m': timedelta(days=60),
            '5m': timedelta(days=60),
            '15m': timedelta(days=60),
            '30m': timedelta(days=60),
            '60m': timedelta(days=730),
            '90m': timedelta(days=730),
            '1h': timedelta(days=730),
        }

        intraday_intervals = ['1m', '2m', '5m', '15m',
                              '30m', '60m', '90m', '1h']
        if self.frequency in intraday_intervals:
            max_period = max_periods.get(
                self.frequency, timedelta(days=60))
            if end_date - start_date > max_period:
                start_date = end_date - max_period
                print(
                    f"Start date set at {start_date.strftime('%Y-%m-%d')} due to limitations for the interval '{self.frequency}'.")

        data = yf.download(
            tickers=self.tickers,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval=self.frequency,
            group_by='ticker',
            auto_adjust=True,
            prepost=False,
            threads=True,
            proxy=None
        )

        if data.empty:
            print("No data taken by Yahoo Finance. Check the tickets and the date interval.")
            return None

        if len(self.tickers) == 1:
            # Just one ticker
            df_close = data[['Close']].rename(
                columns={'Close': self.tickers[0]})
        else:
            # More tickers
            df_close = data['Close']

        return df_close
        
    def loaded_df(self):
        """
        Downloads historical stock price data for the specified time window and tickers using yfinance.
        Returns:
            pandas.df or None: df containing downloaded stock price data if successful, otherwise None.
        """
        if self.years is not None and self.months is None:
            time_window_months = self.years * 12
        elif self.months is not None and self.years is None:
            time_window_months = self.months
        else:
            raise ValueError(
                "Exactly one of 'years' or 'months' should be provided.")

        end_date = datetime.now() - timedelta(days=30)
        start_date = end_date - pd.DateOffset(months=time_window_months)

        stocks_df = self.fetch_yahoo_data(
            start_date=start_date, end_date=end_date)
        if stocks_df is not None:
            PickleHelper(obj=stocks_df).pickle_dump(filename='nasdaq_dataframe')
            return stocks_df
        else:
            print("Impossible to get datas.")
            return None

    def clean_df(self, percentage):
        """
        Cleans the df by dropping stocks with NaN values exceeding the given percentage threshold.
        The cleaned df is pickled after the operation.

        Parameters:
        self
        percentage : float
            Percentage threshold for NaN values. If greater than 1, it's interpreted as a percentage (e.g., 5 for 5%).
        
        Returns:
        None
        """
        if percentage > 1:
            percentage = percentage / 100

        #FIXME: this is not working, it's giving out an empty df
        for ticker in self.tickers:
            nan_values = self.df[ticker].isnull().values.any()
            if nan_values:
                count_nan = self.df[ticker].isnull().sum()
                if count_nan > (len(self.df) * percentage):
                    self.df.drop(ticker, axis=1, inplace=True)

        self.df.fillna(method='ffill', inplace=True)
        #FIXME: fml this doesn't work if i have consecutive days
        PickleHelper(obj=self.df).pickle_dump(filename='cleaned_nasdaq_dataframe')

class Timestamping:
    """
    A class to generate timestamps at specified minute intervals within market trading hours,
    iterating from a start date to an end date. 

    The market trading hours are defined as 9:45 AM to 3:15 PM (based on Simone's hypothesis).
    The generator skips weekends and ensures the timestamps are within trading hours.

    Attributes:
    -----------
    start_date : datetime
        The start date for the timestamp generation.
    end_date : datetime
        The end date for the timestamp generation.
    frequency_minutes : int, optional
        The frequency in minutes for generating timestamps. Default is 1 minute.

    Methods:
    --------
    __iter__():
        Returns the iterator object itself.
    __next__() -> datetime:
        Returns the next timestamp in the sequence.
    """
    def __init__(self, start_date: datetime, end_date: datetime, frequency_minutes=1):
        """
        Initializes the Timestamping class with start date, end date, and frequency in minutes.

        Parameters:
        -----------
        start_date : datetime
            The start date for the timestamp generation.
        end_date : datetime
            The end date for the timestamp generation.
        frequency_minutes : int, optional
            The frequency in minutes for generating timestamps. Default is 1 minute.
        """
        #define market trading hours (from 9:45 to 15:15, considering Simone's hypothesis)
        self.market_open_hour = 9
        self.market_open_minute = 45
        self.market_close_hour = 15
        self.market_close_minute = 15
        #initial assumption: unless stated, starts at the start of the trading day  
        self.current = start_date.replace(hour= self.market_open_hour ,minute=self.market_open_minute -1, second=0, microsecond=0)
        self.end = end_date
        self.frequency = frequency_minutes


    def __iter__(self):
        return self

    def __next__(self) -> datetime:
        """
        Returns the next timestamp in the sequence.

        Increments the current timestamp by the specified frequency (in minutes).
        Adjusts for end-of-day, weekends, and ensures timestamps fall within market trading hours.

        Raises:
        -------
        StopIteration:
            When the current timestamp exceeds the end date.
        
        Returns:
        --------
        datetime
            The next timestamp in the sequence.
        """
        #Increment current to the next stop:
        # - Add 1 minute (later we'll add the frequency here)
        self.current += timedelta(minutes=1)
        # if it's end of day:
        #   - Add 1 day
        if self.current.minute > self.market_close_minute and self.current.hour >= self.market_close_hour:
            self.current += timedelta(days=1)
            self.current = self.current.replace(hour=self.market_open_hour, minute=self.market_open_minute)
        #   if it's end of Friday:
        #     - make it Monday
        if self.current.weekday() == 5:
            self.current += timedelta(days=2)
        if self.current.weekday() == 6:
            self.current += timedelta(days=1)

        if self.current > self.end:
            raise StopIteration

        return self.current

